"""Compare original prompts with reviewed judgments on a fixed fictional corpus."""

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import httpx

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "soupai_benchmark_judging", ROOT / "judging.py"
)
judging = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = judging
spec.loader.exec_module(judging)


class HTTPProvider:
    """Minimal OpenAI-compatible transport for offline-corpus evaluation only."""

    def __init__(self, client, model, extra_body=None):
        self.client = client
        self.model = model
        self.extra_body = extra_body or {}

    async def text_chat(self, *, prompt, system_prompt, **kwargs):
        response = await self.client.post(
            "chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **self.extra_body,
            },
        )
        response.raise_for_status()
        return SimpleNamespace(
            completion_text=response.json()["choices"][0]["message"]["content"] or ""
        )


def baseline_prompts(source, mode, puzzle, answer, content):
    """Read only constant/f-string prompt expressions from the original source.

    Args:
        source: Original plugin source from git or a local backup.
        mode: Question or verification task.
        puzzle: Original puzzle.
        answer: Original solution.
        content: Player submission.

    Returns:
        The original system prompt and user prompt without importing the plugin.
    """
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SoupaiPlugin"
    )
    methods = {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    namespace = {
        "puzzle": puzzle,
        "question": content,
        "true_answer": answer,
        "user_guess": content,
    }
    expressions = []
    if mode == "question":
        method = methods["judge_question"]
        expressions.append(
            next(
                keyword.value
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "text_chat"
                for keyword in node.keywords
                if keyword.arg == "system_prompt"
            )
        )
        expressions.append(
            next(
                node.value
                for node in ast.walk(method)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "prompt"
                    for target in node.targets
                )
            )
        )
    else:
        for name in (
            "_build_verification_system_prompt",
            "_build_verification_user_prompt",
        ):
            expressions.append(
                next(
                    node.value
                    for node in ast.walk(methods[name])
                    if isinstance(node, ast.Return)
                )
            )
    prompts = []
    for expression in expressions:
        if any(
            not isinstance(
                node,
                (
                    ast.JoinedStr,
                    ast.FormattedValue,
                    ast.Constant,
                    ast.Name,
                    ast.Load,
                    ast.BinOp,
                    ast.Add,
                ),
            )
            for node in ast.walk(expression)
        ):
            raise ValueError("Unexpected executable expression in baseline prompt")
        prompts.append(
            eval(
                compile(ast.Expression(expression), "baseline-prompt", "eval"),
                {"__builtins__": {}},
                namespace,
            )
        )
    return prompts


def baseline_label(text, mode):
    """Reproduce legacy parsing, normalizing the two renamed answer labels.

    Args:
        text: Original model completion.
        mode: Question or verification.

    Returns:
        Legacy accepted label, or error for an unparseable completion.
    """
    text = text.strip()
    if mode == "question":
        if text in {"是也不是", "半对半错"}:
            return "是或不是"
        return text if text in judging.ANSWERS else "error"
    level, comment = "", ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("等级："):
            level = line.replace("等级：", "").strip()
        elif line.startswith("评价："):
            comment = line.replace("评价：", "").strip()
    if not level or not comment:
        if "完全还原" in text or "核心推理正确" in text:
            return "核心推理正确" if "核心推理正确" in text else "完全还原"
        return "error"
    return level


async def run(args):
    corpus = json.loads((ROOT / "tests/cases.json").read_text(encoding="utf-8"))
    source = (
        args.baseline_file.read_text(encoding="utf-8")
        if args.baseline_file
        else (
            await asyncio.to_thread(
                subprocess.run,
                ["git", "show", f"{args.baseline_ref}:main.py"],
                cwd=ROOT,
                capture_output=True,
                check=True,
                encoding="utf-8",
            )
        ).stdout
    )
    model = os.environ["SOUPAI_BENCH_MODEL"]
    base_url = os.environ["SOUPAI_BENCH_BASE_URL"].rstrip("/") + "/"
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous["model"] != model:
            raise ValueError("Cannot resume a different model into the same output")
        results = previous["results"]
    completed = {(item["id"], item["variant"]) for item in results}
    headers = {"Authorization": "Bearer " + os.environ["SOUPAI_BENCH_API_KEY"]}
    custom_headers = json.loads(os.environ.get("SOUPAI_BENCH_HEADERS", "{}"))
    headers.update(custom_headers)
    async with httpx.AsyncClient(
        base_url=base_url, headers=headers, timeout=args.timeout
    ) as client:
        provider = HTTPProvider(
            client, model, json.loads(os.environ.get("SOUPAI_BENCH_EXTRA_BODY", "{}"))
        )
        engine = judging.JudgeEngine(provider, timeout=args.timeout)
        tasks = []

        async def evaluate(story, case, mode, case_id, variant, facts):
            if (case_id, variant) in completed:
                return
            async with semaphore:
                started = monotonic()
                calls, actual = 1, "error"
                try:
                    if variant == "baseline":
                        system, prompt = baseline_prompts(
                            source,
                            mode,
                            story["puzzle"],
                            story["answer"],
                            case["content"],
                        )
                        response = await provider.text_chat(
                            prompt=prompt, system_prompt=system
                        )
                        actual = baseline_label(response.completion_text, mode)
                    else:
                        result = await engine.judge(
                            mode=mode,
                            puzzle=story["puzzle"],
                            answer=story["answer"],
                            content=case["content"],
                            history=case.get("history", []),
                            facts=facts,
                        )
                        actual = (
                            result.label if result.status == "ok" else result.status
                        )
                        calls = result.calls
                    error = None
                except Exception as exc:
                    error = type(exc).__name__
                item = {
                    "id": case_id,
                    "variant": variant,
                    "mode": mode,
                    "expected": case["expected"],
                    "actual": actual,
                    "correct": actual == case["expected"],
                    "critical": case.get("critical", False),
                    "calls": calls,
                    "elapsed": round(monotonic() - started, 3),
                    "error": error,
                }
                results.append(item)
                args.output.write_text(
                    json.dumps(
                        {"model": model, "results": results},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(item, ensure_ascii=False), flush=True)

        for story in corpus:
            if args.story and story["id"] not in args.story:
                continue
            # Fact extraction belongs to setup and is shared across this story's cases.
            facts = await engine.prepare_facts(story["puzzle"], story["answer"])
            for mode, key in (
                ("question", "questions"),
                ("verification", "verifications"),
            ):
                for index, case in enumerate(story[key], 1):
                    case_id = f"{story['id']}/{mode}/{index}"
                    for variant in args.variant or ("baseline", "reviewed"):
                        tasks.append(
                            evaluate(story, case, mode, case_id, variant, facts)
                        )
        await asyncio.gather(*tasks)
    summary = {}
    for variant in ("baseline", "reviewed"):
        selected = [item for item in results if item["variant"] == variant]
        if not selected:
            continue
        questions = [item for item in selected if item["mode"] == "question"]
        critical = [item for item in selected if item["critical"]]
        summary[variant] = {
            "correct": sum(item["correct"] for item in selected),
            "total": len(selected),
            "question_accuracy": round(
                sum(item["correct"] for item in questions) / len(questions), 4
            )
            if questions
            else None,
            "critical_correct": sum(item["correct"] for item in critical),
            "critical_total": len(critical),
            "abstentions": sum(
                item["actual"] in {"unknown", "clarify", "error"} for item in selected
            ),
            "calls": sum(item["calls"] for item in selected),
            "mean_seconds": round(
                sum(item["elapsed"] for item in selected) / len(selected), 3
            ),
        }
    args.output.write_text(
        json.dumps(
            {"model": model, "summary": summary, "results": results},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    parser.add_argument("--baseline-ref", default="942d464")
    parser.add_argument("--baseline-file", type=Path)
    parser.add_argument("--story", action="append", help="Evaluate only named stories")
    parser.add_argument("--variant", action="append", choices=("baseline", "reviewed"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(run(parser.parse_args()))
