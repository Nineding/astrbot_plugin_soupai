"""Load the plugin without creating a bot or starting background jobs."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
core = Path(os.environ.get("ASTRBOT_CORE_PATH", ROOT.parent / "core"))
if core.exists():
    sys.path.insert(0, str(core))
spec = importlib.util.spec_from_file_location(
    "soupai_under_test", ROOT / "main.py", submodule_search_locations=[str(ROOT)]
)
plugin_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin_module
spec.loader.exec_module(plugin_module)
judging = sys.modules["soupai_under_test.judging"]


@pytest.fixture
def plugin():
    """Return an uninitialized plugin with isolated in-memory game state."""
    instance = object.__new__(plugin_module.SoupaiPlugin)
    instance.game_state = plugin_module.GameState()
    instance.game_timeout = 1800
    instance.reveal_progress = False
    instance.judge_llm_provider_id = ""
    instance.review_llm_provider_id = ""
    instance.auto_generating = False
    instance.auto_generate_task = None
    instance.auto_schedule_task = None
    instance.generating_games = set()
    instance._session_tasks = set()
    return instance
