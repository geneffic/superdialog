"""Tests for criteria enforcement and skip behavior in DialogStateMachine."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

for _mod in [
    "livekit.agents",
    "livekit.agents.llm",
    "livekit.agents.llm.tool_context",
    "livekit.agents.voice",
    "livekit.api",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402

from superdialog.flow.models import (  # noqa: E402
    CompletionCriterion,
    ConversationFlow,
    Edge,
    FlowNode,
)
from superdialog.machine.machine import DialogStateMachine  # noqa: E402
from superdialog.machine.testing.mock_adapter import (  # noqa: E402
    MockAdapter,
    MockAdapterWithCriteria,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_criteria_flow(allow_skip: bool) -> ConversationFlow:
    """2-node flow with a strict/flexible first node."""
    return ConversationFlow(
        system_prompt="Test",
        initial_node="verify",
        nodes=[
            FlowNode(
                id="verify",
                name="Verify Identity",
                instruction="Ask for ID number.",
                completion_criteria=[
                    CompletionCriterion(
                        key="id_number",
                        description="Customer ID number",
                        required=True,
                    ),
                ],
                allow_skip=allow_skip,
                edges=[
                    Edge(
                        id="verify_to_next",
                        condition="Identity confirmed",
                        target_node_id="next",
                    ),
                ],
            ),
            FlowNode(
                id="next",
                name="Next Step",
                instruction="Proceed.",
                is_final=True,
                edges=[],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# TestStrictNodeBlocksSkip
# ---------------------------------------------------------------------------


class TestStrictNodeBlocksSkip:
    """Tests for criteria enforcement and skip behavior."""

    @pytest.mark.anyio
    async def test_criteria_not_met_no_insistence_stays(self) -> None:
        """criteria_met=False, user_insisting=False, allow_skip=True -> stays."""
        flow = _make_criteria_flow(allow_skip=True)
        adapter = MockAdapterWithCriteria(
            edge_id="verify_to_next",
            criteria_met={"id_number": False},
            user_insisting=False,
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("I don't have my ID")

        assert machine.current_state == "verify"

    @pytest.mark.anyio
    async def test_strict_node_blocks_insistence(self) -> None:
        """criteria_met=False, user_insisting=True, allow_skip=False -> stays."""
        flow = _make_criteria_flow(allow_skip=False)
        adapter = MockAdapterWithCriteria(
            edge_id="verify_to_next",
            criteria_met={"id_number": False},
            user_insisting=True,
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("Just skip it please!")

        assert machine.current_state == "verify"

    @pytest.mark.anyio
    async def test_flexible_node_allows_insistence(self) -> None:
        """criteria_met=False, user_insisting=True, allow_skip=True -> moves."""
        flow = _make_criteria_flow(allow_skip=True)
        adapter = MockAdapterWithCriteria(
            edge_id="verify_to_next",
            criteria_met={"id_number": False},
            user_insisting=True,
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("Just skip it please!")

        assert machine.current_state == "next"
        assert len(machine.context.transition_log) >= 1
        assert machine.context.transition_log[-1].skipped is True

    @pytest.mark.anyio
    async def test_criteria_met_transitions_regardless(self) -> None:
        """criteria_met=True, user_insisting=False, allow_skip=False -> moves."""
        flow = _make_criteria_flow(allow_skip=False)
        adapter = MockAdapterWithCriteria(
            edge_id="verify_to_next",
            criteria_met={"id_number": True},
            user_insisting=False,
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("My ID is 12345")

        assert machine.current_state == "next"
        assert len(machine.context.transition_log) >= 1
        assert machine.context.transition_log[-1].skipped is False


# ---------------------------------------------------------------------------
# TestTransitionSafety
# ---------------------------------------------------------------------------


class TestTransitionSafety:
    """Tests for transition safety and edge validation."""

    @pytest.mark.anyio
    async def test_invalid_edge_from_state_stays_put(self) -> None:
        """3-node flow (a->b->c), fire 'b_to_c' from state 'a' -> stays."""
        flow = ConversationFlow(
            system_prompt="Test",
            initial_node="a",
            nodes=[
                FlowNode(
                    id="a",
                    name="Node A",
                    instruction="Do A.",
                    edges=[
                        Edge(
                            id="a_to_b",
                            condition="go to b",
                            target_node_id="b",
                        ),
                    ],
                ),
                FlowNode(
                    id="b",
                    name="Node B",
                    instruction="Do B.",
                    edges=[
                        Edge(
                            id="b_to_c",
                            condition="go to c",
                            target_node_id="c",
                        ),
                    ],
                ),
                FlowNode(
                    id="c",
                    name="Node C",
                    instruction="Done.",
                    is_final=True,
                    edges=[],
                ),
            ],
        )
        # Adapter recommends b_to_c, but we're at state "a"
        adapter = MockAdapter(edge_sequence=["b_to_c"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("try wrong edge")

        assert machine.current_state == "a"

    @pytest.mark.anyio
    async def test_already_complete_no_transition(self) -> None:
        """After reaching final node, process_turn does nothing."""
        flow = _make_criteria_flow(allow_skip=False)
        adapter = MockAdapterWithCriteria(
            edge_id="verify_to_next",
            criteria_met={"id_number": True},
            user_insisting=False,
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # First turn: transition to final node
        await machine.process_turn("My ID is 12345")
        assert machine.current_state == "next"
        assert machine.is_complete is True
        log_length = len(machine.context.transition_log)

        # Second turn: should be a no-op
        await machine.process_turn("anything else")
        assert len(machine.context.transition_log) == log_length
