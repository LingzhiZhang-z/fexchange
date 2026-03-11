from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "codex_mcp_router.py"


def _load_router_module():
    spec = importlib.util.spec_from_file_location("codex_mcp_router", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mode_registry_contains_exactly_four_modes():
    router = _load_router_module()
    assert set(router.MODE_CONFIGS) == {
        "high",
        "xhigh",
        "high_fast",
        "xhigh_fast",
    }


def test_high_and_xhigh_build_distinct_reasoning_commands():
    router = _load_router_module()
    high = router.build_codex_command("high", "say hi", cwd="/tmp")
    xhigh = router.build_codex_command("xhigh", "say hi", cwd="/tmp")

    assert "high" in high
    assert "xhigh" in xhigh
    assert high != xhigh


def test_fast_modes_are_distinct_from_non_fast_modes():
    router = _load_router_module()
    high = router.build_codex_command("high", "say hi")
    high_fast = router.build_codex_command("high_fast", "say hi")

    assert high != high_fast
    assert "high" in high
    assert "high_fast" in high_fast


def test_build_subprocess_kwargs_forwards_cwd():
    router = _load_router_module()
    kwargs = router.build_subprocess_kwargs("/tmp/router-test")
    assert kwargs["cwd"] == "/tmp/router-test"
    assert kwargs["text"] is True


def test_build_tool_definitions_exposes_four_codex_tools():
    router = _load_router_module()
    tools = router.build_tool_definitions()
    assert {tool["name"] for tool in tools} == {
        "codex_high",
        "codex_xhigh",
        "codex_high_fast",
        "codex_xhigh_fast",
    }


def test_dispatch_tool_call_rejects_missing_prompt():
    router = _load_router_module()
    with pytest.raises(ValueError, match="prompt"):
        router.dispatch_tool_call("codex_high", {})
