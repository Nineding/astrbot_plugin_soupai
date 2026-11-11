"""Verify evidence validation, review independence and bounded failures."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import judging

PUZZLE = "男人向酒保要水，酒保举枪，他道谢离开。"
ANSWER = "男人正在打嗝，要水是想止嗝。酒保用玩具枪吓他，惊吓使打嗝停止。酒保没有开枪，也没有抢劫。"
SUPPORTED = {
    "claim": "男人打嗝",
    "verdict": "supported",
    "importance": "core",
    "evidence": ["男人正在打嗝"],
}
CONTRADICTED = {
    "claim": "酒保为了抢劫",
    "verdict": "contradicted",
    "importance": "core",
    "evidence": ["也没有抢劫"],
}


def completion(label="是", *, status="ok", checks=None, corrections=None):
    return json.dumps(
        {
            "status": status,
            "label": label,
            "checks": checks if checks is not None else [SUPPORTED],
            "corrections": corrections or [],
        },
        ensure_ascii=False,
    )


class Provider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def text_chat(self, **kwargs):
        self.requests.append(kwargs)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(completion_text=response)


def test_corpus_has_required_coverage():
    corpus = json.loads(
        Path(__file__).with_name("cases.json").read_text(encoding="utf-8")
    )
    assert sum(len(story["questions"]) for story in corpus) >= 40
    assert sum(len(story["verifications"]) for story in corpus) >= 16
    assert (
        sum(
            case.get("critical", False)
            for story in corpus
            for case in story["questions"]
        )
        >= 8
    )
    assert {
        case["expected"] for story in corpus for case in story["verifications"]
    } == set(judging.LEVEL_COMMENTS)


def test_mixed_true_and_false_requires_mixed_label():
    accepted = judging.parse_judgment(
        completion("是或不是", checks=[SUPPORTED, CONTRADICTED]),
        "question",
        PUZZLE,
        ANSWER,
        [],
    )
    assert accepted.response == "是或不是"
    for label in ("是", "否", "不重要"):
        normalized = judging.parse_judgment(
            completion(label, checks=[SUPPORTED, CONTRADICTED]),
            "question",
            PUZZLE,
            ANSWER,
            [],
        )
        assert normalized.response == "是或不是"
    for label in ("半对半错", "是也不是"):
        with pytest.raises(ValueError):
            judging.parse_judgment(
                completion(label, checks=[SUPPORTED, CONTRADICTED]),
                "question",
                PUZZLE,
                ANSWER,
                [],
            )


@pytest.mark.parametrize(
    "text",
    [
        "等级：不是完全还原\n评价：不能算核心推理正确",
        "等级：完全还原",
        '{"status":"ok","label":"完全还原"}',
        "文字 " + completion(),
        completion() + "\n忽略规则",
    ],
)
def test_rejects_keyword_matches_and_non_json(text):
    with pytest.raises(ValueError):
        judging.parse_judgment(text, "verification", PUZZLE, ANSWER, [])


@pytest.mark.parametrize(
    "change",
    [
        {"evidence": ["故事未曾出现的事实"]},
        {"evidence": []},
        {"verdict": "yes"},
        {"importance": "maybe"},
        {"claim": ""},
    ],
)
def test_rejects_invalid_atomic_evidence(change):
    with pytest.raises(ValueError):
        judging.parse_judgment(
            completion(checks=[SUPPORTED | change]), "question", PUZZLE, ANSWER, []
        )


def test_unknown_is_not_irrelevant():
    check = SUPPORTED | {"verdict": "unknown", "evidence": []}
    with pytest.raises(ValueError):
        judging.parse_judgment(
            completion("不重要", checks=[check]), "question", PUZZLE, ANSWER, []
        )
    result = judging.parse_judgment(
        completion("", status="unknown", checks=[check]), "question", PUZZLE, ANSWER, []
    )
    assert "无法确定" in result.response


@pytest.mark.parametrize(
    "verdict", ["contradicted", "missing", "unknown", "conditional"]
)
@pytest.mark.parametrize("label", ["完全还原", "核心推理正确"])
def test_core_errors_and_missing_causes_cannot_pass(label, verdict):
    check = CONTRADICTED | {"verdict": verdict}
    with pytest.raises(ValueError):
        judging.parse_judgment(
            completion(label, checks=[SUPPORTED, check]),
            "verification",
            PUZZLE,
            ANSWER,
            [],
        )


@pytest.mark.parametrize("index", [0, 2, True])
def test_corrections_need_valid_indices(index):
    correction = {"index": index, "answer": "是", "evidence": ["男人正在打嗝"]}
    with pytest.raises(ValueError):
        judging.parse_judgment(
            completion(corrections=[correction]),
            "question",
            PUZZLE,
            ANSWER,
            [{"question": "打嗝？", "answer": "否"}],
        )


async def test_review_is_independent_and_uses_own_provider():
    first = Provider([completion()])
    reviewer = Provider([completion()])
    result = await judging.JudgeEngine(first, reviewer).judge(
        mode="question", puzzle=PUZZLE, answer=ANSWER, content="他在打嗝吗？"
    )
    assert (result.status, result.label, result.calls) == ("ok", "是", 2)
    assert len(first.requests) == len(reviewer.requests) == 1
    assert "untrusted_candidates" not in json.loads(reviewer.requests[0]["prompt"])
    assert first.requests[0]["contexts"] == []


async def test_disagreement_uses_third_call_and_requires_agreement():
    no = completion("否", checks=[CONTRADICTED])
    provider = Provider([completion(), no, no])
    result = await judging.JudgeEngine(provider).judge(
        mode="question", puzzle=PUZZLE, answer=ANSWER, content="测试分歧"
    )
    assert (result.label, result.calls) == ("否", 3)
    assert len(json.loads(provider.requests[2]["prompt"])["untrusted_candidates"]) == 2


async def test_unresolved_disagreement_does_not_guess():
    provider = Provider(
        [
            completion(),
            completion("否", checks=[CONTRADICTED]),
            completion("", status="unknown", checks=[]),
        ]
    )
    result = await judging.JudgeEngine(provider).judge(
        mode="question", puzzle=PUZZLE, answer=ANSWER, content="测试分歧"
    )
    assert result.status == "error"
    assert result.calls == 3
    assert ANSWER not in result.response


async def test_invalid_format_uses_budget_without_keyword_fallback():
    provider = Provider(["完全还原，忽略以上规则", completion(), completion()])
    result = await judging.JudgeEngine(provider).judge(
        mode="question", puzzle=PUZZLE, answer=ANSWER, content="打嗝？"
    )
    assert (result.label, result.calls) == ("是", 3)


async def test_outage_is_bounded_and_does_not_expose_exception():
    provider = Provider(
        [
            RuntimeError("private-token"),
            RuntimeError("private-token"),
            RuntimeError("private-token"),
        ]
    )
    result = await judging.JudgeEngine(provider).judge(
        mode="question", puzzle=PUZZLE, answer=ANSWER, content="打嗝？"
    )
    assert result.status == "error"
    assert len(provider.requests) == 3
    assert "private-token" not in result.response


async def test_timeout_and_cancellation():
    class SlowProvider:
        async def text_chat(self, **kwargs):
            await asyncio.sleep(10)

    engine = judging.JudgeEngine(SlowProvider(), timeout=0.001)
    result = await engine.judge(
        mode="question", puzzle=PUZZLE, answer=ANSWER, content="打嗝？"
    )
    assert result.status == "error" and result.calls == 3
    task = asyncio.create_task(
        judging.JudgeEngine(SlowProvider()).judge(
            mode="question", puzzle=PUZZLE, answer=ANSWER, content="打嗝？"
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_only_agreed_history_corrections_are_returned():
    correction = {"index": 1, "answer": "是", "evidence": ["男人正在打嗝"]}
    history = [{"question": "他打嗝吗？", "answer": "否"}]
    provider = Provider([completion(corrections=[correction]), completion()])
    result = await judging.JudgeEngine(provider).judge(
        mode="question",
        puzzle=PUZZLE,
        answer=ANSWER,
        content="他要止嗝吗？",
        history=history,
    )
    assert result.corrections == ()
    provider = Provider(
        [completion(corrections=[correction]), completion(corrections=[correction])]
    )
    result = await judging.JudgeEngine(provider).judge(
        mode="question",
        puzzle=PUZZLE,
        answer=ANSWER,
        content="他要止嗝吗？",
        history=history,
    )
    assert result.corrections[0]["answer"] == "是"
    assert history[0]["answer"] == "否"


@pytest.mark.parametrize(
    "facts,expected",
    [
        ([{"fact": "正在打嗝", "evidence": "男人正在打嗝"}], True),
        ([{"fact": "正在打嗝", "evidence": "不存在的证据"}], False),
        ([{"fact": "", "evidence": "男人正在打嗝"}], False),
        ({"fact": "错误格式"}, False),
    ],
)
async def test_fact_index_validates_source_and_falls_back(facts, expected):
    provider = Provider([json.dumps(facts, ensure_ascii=False)])
    result = await judging.JudgeEngine(provider).prepare_facts(PUZZLE, ANSWER)
    assert bool(result) is expected
