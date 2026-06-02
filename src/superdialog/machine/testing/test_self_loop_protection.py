# flake8: noqa
"""Tests for self-loop protection in the dialog state machine.

Covers three protections:
  A) Gate denies self-loop edges after MAX_SELF_LOOPS consecutive loops.
  B) Static text is not re-spoken on self-loop transitions.
  C) Self-targeting edges are filtered from tools once user has responded.
"""

from __future__ import annotations

# Allow running without the full livekit/pipecat SDK installed.
# Each module needs its own MagicMock instance to avoid the
# "'X' is not a package" error from Python's import machinery.
# Dynamically discover and mock livekit.* and pipecat.* modules.
import importlib.abc
import importlib.machinery
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


def _make_module_mock(fullname: str) -> MagicMock:
    """Create a MagicMock that behaves like a Python package/module."""
    mock = MagicMock()
    mock.__name__ = fullname
    mock.__path__ = []
    mock.__package__ = fullname
    mock.__spec__ = importlib.machinery.ModuleSpec(fullname, None)
    mock.__file__ = f"<mock {fullname}>"
    return mock


class _AutoMockFinder(importlib.abc.MetaPathFinder):
    """Auto-mock any livekit.* or pipecat.* import."""

    _PREFIXES = ("livekit", "pipecat")

    def find_module(self, fullname: str, path: Any = None) -> "_AutoMockFinder | None":
        for prefix in self._PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return self
        return None

    def load_module(self, fullname: str) -> MagicMock:
        if fullname in sys.modules:
            return sys.modules[fullname]
        mock = _make_module_mock(fullname)
        sys.modules[fullname] = mock
        return mock


sys.meta_path.insert(0, _AutoMockFinder())

from superdialog.flow.models import ConversationFlow
from superdialog.machine.machine import DialogStateMachine
from superdialog.machine.models import (
    FlowContext,
    NodeScope,
    ToolDescriptor,
    TransitionResult,
)
from superdialog.machine.testing.mock_adapter import MockAdapter

# ── Fixtures ────────────────────────────────────────────────────────────

SELF_LOOP_FLOW = {
    "system_prompt": "Test assistant.",
    "initial_node": "ask_question",
    "agent_language": "en",
    "agent_gender": "female",
    "global_edges": [],
    "nodes": [
        {
            "id": "ask_question",
            "name": "Ask Question",
            "static_text": "Did you receive the card?",
            "interruptible": True,
            "edges": [
                {
                    "id": "edge_yes",
                    "condition": "User says yes",
                    "target_node_id": "done",
                },
                {
                    "id": "edge_no",
                    "condition": "User says no",
                    "target_node_id": "done",
                },
                {
                    "id": "edge_interrupt",
                    "condition": "User interrupted during speech",
                    "target_node_id": "ask_question",
                },
            ],
        },
        {
            "id": "done",
            "name": "Done",
            "static_text": "Thank you!",
            "is_final": True,
        },
    ],
    "actions": [],
}


def _write_flow(flow_dict: dict[str, Any]) -> str:
    """Write a flow dict to a temp JSON file, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(flow_dict, tmp)
    tmp.close()
    return tmp.name


@pytest.fixture
def flow_path() -> str:
    return _write_flow(SELF_LOOP_FLOW)


async def _build_machine(
    flow_path: str,
    edge_sequence: list[str] | None = None,
) -> tuple[DialogStateMachine, MockAdapter]:
    """Build a DialogStateMachine from a flow file."""
    flow = ConversationFlow.from_file(flow_path)
    adapter = MockAdapter(edge_sequence=edge_sequence or [])
    machine = await DialogStateMachine.from_flow(flow, adapter)
    return machine, adapter


# ── Test A: Gate denies after MAX_SELF_LOOPS ─────────────────────────


@pytest.mark.anyio
async def test_gate_denies_after_max_self_loops(flow_path: str) -> None:
    """After MAX_SELF_LOOPS self-loops, the gate should deny the
    self-loop edge and return a correction hint."""
    machine, adapter = await _build_machine(flow_path)

    assert machine.state == "ask_question"

    # Simulate: mark node as spoken + user has responded
    machine.mark_node_spoken()
    machine.context.add_user_message("hello")

    # First self-loop should succeed
    result1 = await machine.request_transition("edge_interrupt")
    assert result1.allowed is True
    assert machine.state == "ask_question"

    # After first loop, consecutive_self_loops == 1
    assert machine.context.consecutive_self_loops == 1

    # Prepare for second attempt: mark spoken + user spoke
    machine.mark_node_spoken()
    machine.context.add_user_message("yes I got it")

    # Second self-loop should succeed (at limit but not over)
    result2 = await machine.request_transition("edge_interrupt")
    assert result2.allowed is True
    assert machine.context.consecutive_self_loops == 2

    # Prepare for third attempt
    machine.mark_node_spoken()
    machine.context.add_user_message("I said yes")

    # Third self-loop should be DENIED (>= MAX_SELF_LOOPS)
    result3 = await machine.request_transition("edge_interrupt")
    assert result3.allowed is False
    assert "self-loop limit" in (result3.reason or "").lower()
    assert result3.correction_hint != ""

    # Non-self-loop edge should still work
    result4 = await machine.request_transition("edge_yes")
    assert result4.allowed is True
    assert machine.state == "done"


@pytest.mark.anyio
async def test_self_loop_counter_resets_on_different_target(
    flow_path: str,
) -> None:
    """Transitioning to a different node resets the counter."""
    machine, adapter = await _build_machine(flow_path)

    machine.mark_node_spoken()
    machine.context.add_user_message("hello")

    # Do one self-loop
    result = await machine.request_transition("edge_interrupt")
    assert result.allowed is True
    assert machine.context.consecutive_self_loops == 1

    # Now transition to a different node
    machine.mark_node_spoken()
    machine.context.add_user_message("yes")
    result2 = await machine.request_transition("edge_yes")
    assert result2.allowed is True
    assert machine.state == "done"
    assert machine.context.consecutive_self_loops == 0


# ── Test B: NodeScope.is_self_loop flag ─────────────────────────────


@pytest.mark.anyio
async def test_node_scope_is_self_loop_flag(flow_path: str) -> None:
    """After a self-loop, build_node_scope() should set is_self_loop=True."""
    machine, adapter = await _build_machine(flow_path)

    # Before any self-loop
    scope_before = machine.build_node_scope()
    assert scope_before.is_self_loop is False

    # Trigger a self-loop
    machine.mark_node_spoken()
    machine.context.add_user_message("what?")
    result = await machine.request_transition("edge_interrupt")
    assert result.allowed is True

    # After self-loop
    scope_after = machine.build_node_scope()
    assert scope_after.is_self_loop is True


# ── Test C: Self-targeting edges filtered from tools ────────────────


@pytest.mark.anyio
async def test_self_targeting_edges_filtered_after_user_responds(
    flow_path: str,
) -> None:
    """Once user has spoken AND node is spoken, self-targeting edges
    should not appear in tools."""
    machine, adapter = await _build_machine(flow_path)

    # Initially, all edges should be present (including self-loop)
    tools_initial = machine.get_tools_for_node()
    tool_ids_initial = {t.id for t in tools_initial}
    assert "edge_interrupt" in tool_ids_initial
    assert "edge_yes" in tool_ids_initial
    assert "edge_no" in tool_ids_initial

    # Mark node as spoken but user hasn't spoken yet
    machine.mark_node_spoken()
    tools_spoken = machine.get_tools_for_node()
    tool_ids_spoken = {t.id for t in tools_spoken}
    # Self-loop should still be there (user hasn't responded)
    assert "edge_interrupt" in tool_ids_spoken

    # Now user speaks
    machine.context.add_user_message("I don't know")
    tools_post_user = machine.get_tools_for_node()
    tool_ids_post_user = {t.id for t in tools_post_user}

    # Self-targeting edge should be filtered OUT
    assert "edge_interrupt" not in tool_ids_post_user
    # Other edges should remain
    assert "edge_yes" in tool_ids_post_user
    assert "edge_no" in tool_ids_post_user


@pytest.mark.anyio
async def test_non_self_targeting_edges_not_filtered(
    flow_path: str,
) -> None:
    """Edges that target a different node should never be filtered."""
    machine, adapter = await _build_machine(flow_path)

    machine.mark_node_spoken()
    machine.context.add_user_message("yes")

    tools = machine.get_tools_for_node()
    tool_ids = {t.id for t in tools}
    assert "edge_yes" in tool_ids
    assert "edge_no" in tool_ids


# ── Test: Combined flow ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_full_self_loop_protection_flow(flow_path: str) -> None:
    """End-to-end: self-loop happens, tools filter kicks in, forward
    transition still works."""
    machine, adapter = await _build_machine(flow_path)

    # Turn 1: node spoken, user responds, LLM picks interrupt
    machine.mark_node_spoken()
    machine.context.add_user_message("hello?")

    # After user speaks, interrupt edge should be filtered from tools
    tools = machine.get_tools_for_node()
    tool_ids = {t.id for t in tools}
    assert "edge_interrupt" not in tool_ids

    # But if the gate is called anyway (e.g., race condition),
    # the first self-loop would still pass gate check
    result = await machine.request_transition("edge_interrupt")
    assert result.allowed is True
    assert machine.context.consecutive_self_loops == 1

    # After self-loop, on next scope build, is_self_loop is True
    scope = machine.build_node_scope()
    assert scope.is_self_loop is True

    # Forward transition always works
    machine.mark_node_spoken()
    machine.context.add_user_message("yes")
    result2 = await machine.request_transition("edge_yes")
    assert result2.allowed is True
    assert machine.state == "done"
    assert machine.context.consecutive_self_loops == 0


# ── Test: Flow control block for instruction nodes ──────────────────

INSTRUCTION_NODE_FLOW = {
    "system_prompt": "Test assistant.",
    "initial_node": "echo_confirm",
    "agent_language": "en",
    "agent_gender": "female",
    "global_edges": [],
    "nodes": [
        {
            "id": "echo_confirm",
            "name": "Echo Confirm",
            "instruction": (
                "Repeat back the year of birth and ask the caller to confirm."
            ),
            "interruptible": True,
            "edges": [
                {
                    "id": "edge_confirmed",
                    "condition": "User confirms",
                    "target_node_id": "done",
                },
            ],
        },
        {
            "id": "done",
            "name": "Done",
            "static_text": "Thank you!",
            "is_final": True,
        },
    ],
    "actions": [],
}

INSTRUCTION_WITH_MARKERS_FLOW = {
    "system_prompt": "Test assistant.",
    "initial_node": "echo_confirm",
    "agent_language": "en",
    "agent_gender": "female",
    "global_edges": [],
    "nodes": [
        {
            "id": "echo_confirm",
            "name": "Echo Confirm",
            "instruction": (
                "[EN] Your year of birth is 1990, is that correct?\n"
                "[HI] Aapka janam varsh 1990 hai, kya yeh sahi hai?"
            ),
            "interruptible": True,
            "edges": [
                {
                    "id": "edge_confirmed",
                    "condition": "User confirms",
                    "target_node_id": "done",
                },
            ],
        },
        {
            "id": "done",
            "name": "Done",
            "static_text": "Thank you!",
            "is_final": True,
        },
    ],
    "actions": [],
}


@pytest.mark.anyio
async def test_flow_control_block_no_markers_says_speak_first() -> None:
    """For instruction nodes WITHOUT language markers (generate_reply),
    the flow control block should say 'SPEAK' not 'already spoken'."""
    path = _write_flow(INSTRUCTION_NODE_FLOW)
    machine, _ = await _build_machine(path)

    enriched = machine.get_enriched_instructions()
    # Should NOT say "already been spoken via text-to-speech"
    assert "already been spoken" not in enriched
    # Should tell LLM to speak first
    assert "SPEAK" in enriched
    # Should tell LLM not to call tools in this response
    assert "Do NOT call any transition tool in this response" in enriched


@pytest.mark.anyio
async def test_flow_control_block_with_markers_says_already_spoken() -> None:
    """For instruction nodes WITH language markers (session.say),
    the flow control block should say 'already spoken'."""
    path = _write_flow(INSTRUCTION_WITH_MARKERS_FLOW)
    machine, _ = await _build_machine(path)

    enriched = machine.get_enriched_instructions()
    # Should say "already spoken" since TTS handles it
    assert "already been spoken" in enriched
    # Should tell LLM to WAIT
    assert "WAIT" in enriched


# ── Test: Gate denial includes recovery_speech ──────────────────────


@pytest.mark.anyio
async def test_gate_4a_denial_has_recovery_speech() -> None:
    """Gate 4A (no user input) should return a recovery_speech
    so the caller hears something instead of silence."""
    # Gate 4A only applies to instruction/router nodes, not static
    path = _write_flow(INSTRUCTION_NODE_FLOW)
    machine, _ = await _build_machine(path)

    # Mark node spoken but don't add user message
    machine.mark_node_spoken()

    result = await machine.request_transition("edge_confirmed")
    assert result.allowed is False
    assert "no user input" in result.reason.lower()
    assert result.recovery_speech != ""
    assert "listening" in result.recovery_speech.lower()


@pytest.mark.anyio
async def test_gate_3_denial_has_recovery_speech() -> None:
    """Gate 3 (missing criteria) should return a recovery_speech."""
    criteria_flow = {
        "system_prompt": "Test.",
        "initial_node": "collect",
        "agent_language": "en",
        "agent_gender": "female",
        "global_edges": [],
        "nodes": [
            {
                "id": "collect",
                "name": "Collect",
                "instruction": "Collect name.",
                "interruptible": True,
                "completion_criteria": [
                    {
                        "key": "name",
                        "description": "Full name",
                        "required": True,
                    }
                ],
                "allow_skip": False,
                "edges": [
                    {
                        "id": "edge_done",
                        "condition": "Name collected",
                        "target_node_id": "end",
                    }
                ],
            },
            {
                "id": "end",
                "name": "End",
                "static_text": "Done.",
                "is_final": True,
            },
        ],
        "actions": [],
    }
    path = _write_flow(criteria_flow)
    machine, _ = await _build_machine(path)

    machine.mark_node_spoken()
    machine.context.add_user_message("hello")

    result = await machine.request_transition("edge_done")
    assert result.allowed is False
    assert result.recovery_speech != ""
    assert "details" in result.recovery_speech.lower()


@pytest.mark.anyio
async def test_gate_denial_no_interrupt_preserves_speech(
    flow_path: str,
) -> None:
    """Gate denial should NOT set recovery_speech to empty for
    gates that don't need caller communication (Gate 1, Gate 2)."""
    machine, _ = await _build_machine(flow_path)

    # Gate 1: invalid edge — no recovery speech needed
    result = await machine.request_transition("nonexistent_edge")
    assert result.allowed is False
    assert result.recovery_speech == ""

    # Gate 2: node not spoken — no recovery speech needed
    result2 = await machine.request_transition("edge_yes")
    assert result2.allowed is False
    assert result2.recovery_speech == ""
