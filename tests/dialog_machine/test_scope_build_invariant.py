"""Regression tests for the scope-build-once invariant.

Locks the contract that ``DialogStateMachine.build_node_scope`` runs
exactly once per transition on the gated path, and that
``FlowExecutor.handle_transition(scope=...)`` reuses the gate-built
scope without rebuilding.

See `openspec/changes/eliminate-duplicate-scope-build/`.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

# Mock LiveKit SDK so the agent modules import without the real SDK.
for _mod in [
    "livekit",
    "livekit.agents",
    "livekit.agents.llm",
    "livekit.agents.llm.tool_context",
    "livekit.agents.voice",
    "livekit.api",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402

from superdialog.flow.models import ConversationFlow, Edge, FlowNode  # noqa: E402
from superdialog.machine.machine import DialogStateMachine  # noqa: E402
from superdialog.machine.models import NodeScope  # noqa: E402
from superdialog.machine.testing.mock_adapter import MockAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_node_flow() -> ConversationFlow:
    """Minimal start→end flow with one transition edge."""
    return ConversationFlow(
        system_prompt="Test agent.",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                static_text="Hello!",
                edges=[
                    Edge(
                        id="e_start_end",
                        condition="user ready",
                        target_node_id="end",
                    ),
                ],
            ),
            FlowNode(
                id="end",
                name="End",
                static_text="Goodbye!",
                is_final=True,
            ),
        ],
    )


class _BuildCounter:
    """Wraps ``DialogStateMachine.build_node_scope`` to count invocations."""

    def __init__(self, machine: DialogStateMachine) -> None:
        self.machine = machine
        self.calls = 0
        self._original = machine.build_node_scope

        def counted(node: Any = None) -> NodeScope:
            self.calls += 1
            return self._original(node)

        machine.build_node_scope = counted  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Test 1 — gate path builds scope exactly once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_transition_builds_scope_once_per_call() -> None:
    """request_transition must call build_node_scope exactly once
    per allowed transition (for the new node's scope)."""
    flow = _two_node_flow()
    adapter = MockAdapter(edge_sequence=[])
    machine = await DialogStateMachine.from_flow(flow, adapter)

    # Gate requires the current node to be marked spoken.
    machine.mark_node_spoken("start")
    machine.context.user_turns_in_node = 1

    counter = _BuildCounter(machine)

    result = await machine.request_transition("e_start_end")
    assert result.allowed, f"gate denied: {result.reason}"
    assert result.new_scope is not None
    assert result.new_scope.node_id == "end"
    assert (
        counter.calls == 1
    ), f"expected exactly 1 build_node_scope call, got {counter.calls}"


# ---------------------------------------------------------------------------
# Test 2 — handle_transition with pre-built scope skips rebuild
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_transition_with_scope_skips_rebuild() -> None:
    """When FlowExecutor.handle_transition is called with scope=...,
    it MUST NOT call build_node_scope."""
    from superdialog.machine.adapters.flow_executor import FlowExecutor

    flow = _two_node_flow()
    adapter = MockAdapter(edge_sequence=[])
    machine = await DialogStateMachine.from_flow(flow, adapter)
    machine.mark_node_spoken("start")
    machine.context.user_turns_in_node = 1

    # Gate produces a real scope.
    gate = await machine.request_transition("e_start_end")
    assert gate.allowed and gate.new_scope is not None
    pre_built_scope = gate.new_scope

    # Now drive the executor.
    state = MagicMock()
    state.mark_processing = MagicMock(return_value=None)

    executor = FlowExecutor(machine=machine, state=state, config={})

    # Mock agent + session
    fake_session = MagicMock()
    fake_session.interrupt = MagicMock(return_value=None)
    fake_session.update_agent = MagicMock(return_value=None)
    fake_agent = MagicMock()
    fake_agent.session = fake_session

    # Patch create_agent_for_scope so we don't need real LiveKit Agent classes.
    sentinel_new_agent = MagicMock(name="new_agent")
    executor.create_agent_for_scope = MagicMock(return_value=sentinel_new_agent)  # type: ignore[method-assign]

    counter = _BuildCounter(machine)

    await executor.handle_transition(
        edge_id="e_start_end",
        turn_result=gate.turn_result,
        agent=fake_agent,
        scope=pre_built_scope,
    )

    assert counter.calls == 0, (
        f"handle_transition must not rebuild when scope is provided; "
        f"got {counter.calls} calls"
    )
    executor.create_agent_for_scope.assert_called_once_with(pre_built_scope)
    fake_session.update_agent.assert_called_once_with(sentinel_new_agent)


# ---------------------------------------------------------------------------
# Test 3 — no `from_node` write to node_spoken_flags after transition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_from_node_spoken_write_after_transition() -> None:
    """After handle_transition runs, the executor must NOT have written
    node_spoken_flags[from_node]. The flag was set only by the prior
    on_enter; the executor adds nothing new for from_node."""
    from superdialog.machine.adapters.flow_executor import FlowExecutor

    flow = _two_node_flow()
    adapter = MockAdapter(edge_sequence=[])
    machine = await DialogStateMachine.from_flow(flow, adapter)

    # Simulate the old agent's on_enter: it sets the flag for 'start'.
    machine.mark_node_spoken("start")
    machine.context.user_turns_in_node = 1

    snapshot_before = dict(machine.context.node_spoken_flags)
    assert snapshot_before == {"start": True}

    gate = await machine.request_transition("e_start_end")
    assert gate.allowed

    state = MagicMock()
    state.mark_processing = MagicMock(return_value=None)
    executor = FlowExecutor(machine=machine, state=state, config={})

    fake_session = MagicMock()
    fake_session.interrupt = MagicMock(return_value=None)
    fake_session.update_agent = MagicMock(return_value=None)
    fake_agent = MagicMock()
    fake_agent.session = fake_session
    executor.create_agent_for_scope = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

    await executor.handle_transition(
        edge_id="e_start_end",
        turn_result=gate.turn_result,
        agent=fake_agent,
        scope=gate.new_scope,
    )

    # The executor must not have added any new key. The only valid
    # post-state is the same set of keys we had before (the new agent's
    # on_enter, which is mocked here, is what would set 'end'=True in
    # production — not the executor).
    assert machine.context.node_spoken_flags == snapshot_before, (
        f"executor wrote unexpected node_spoken_flags entries; "
        f"before={snapshot_before} after={machine.context.node_spoken_flags}"
    )


# ---------------------------------------------------------------------------
# Test 5 — enrichment runs at most once per transition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_scope_called_once_per_transition() -> None:
    """request_transition must call _enrich_scope exactly once for the
    new scope. The executor (when handed a pre-enriched scope) must
    NOT call _enrich_scope again."""
    from superdialog.machine.adapters.flow_executor import FlowExecutor

    flow = _two_node_flow()
    adapter = MockAdapter(edge_sequence=[])
    machine = await DialogStateMachine.from_flow(flow, adapter)
    machine.mark_node_spoken("start")
    machine.context.user_turns_in_node = 1

    enrich_calls = {"n": 0}
    original_enrich = machine._enrich_scope

    async def counting_enrich(scope: NodeScope) -> NodeScope:
        enrich_calls["n"] += 1
        return await original_enrich(scope)

    machine._enrich_scope = counting_enrich  # type: ignore[method-assign]

    gate = await machine.request_transition("e_start_end")
    assert gate.allowed
    assert (
        enrich_calls["n"] == 1
    ), f"expected 1 enrich call from gate, got {enrich_calls['n']}"

    state = MagicMock()
    state.mark_processing = MagicMock(return_value=None)
    executor = FlowExecutor(machine=machine, state=state, config={})
    fake_session = MagicMock()
    fake_session.interrupt = MagicMock(return_value=None)
    fake_session.update_agent = MagicMock(return_value=None)
    fake_agent = MagicMock()
    fake_agent.session = fake_session
    executor.create_agent_for_scope = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

    await executor.handle_transition(
        edge_id="e_start_end",
        turn_result=gate.turn_result,
        agent=fake_agent,
        scope=gate.new_scope,
    )

    assert enrich_calls["n"] == 1, (
        f"executor must NOT re-enrich when scope is supplied; "
        f"got {enrich_calls['n']} total enrich calls"
    )
