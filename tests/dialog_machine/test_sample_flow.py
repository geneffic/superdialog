"""Tests using the sample appointment flow JSON.

Demonstrates how to load a flow from JSON and drive it through
various paths using MockAdapter.
"""

from __future__ import annotations

import sys
from pathlib import Path
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

from superdialog.flow.models import ConversationFlow  # noqa: E402
from superdialog.machine.machine import DialogStateMachine  # noqa: E402
from superdialog.machine.testing.mock_adapter import (  # noqa: E402
    MockAdapter,
    MockAdapterWithCriteria,
)

SAMPLE_FLOW_PATH = (
    Path(__file__).resolve().parents[4]
    / "super"
    / "core"
    / "voice"
    / "dialog_machine"
    / "testing"
    / "sample_appointment_flow.json"
)


@pytest.fixture
def flow() -> ConversationFlow:
    return ConversationFlow.from_json_file(SAMPLE_FLOW_PATH)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


class TestFlowStructure:
    """Verify the sample flow loads correctly and has expected shape."""

    def test_loads_from_json(self, flow: ConversationFlow) -> None:
        assert flow.initial_node == "greeting"
        assert len(flow.nodes) == 5

    def test_node_ids(self, flow: ConversationFlow) -> None:
        ids = {n.id for n in flow.nodes}
        assert ids == {"greeting", "collect_info", "faq", "confirm", "goodbye"}

    def test_final_node(self, flow: ConversationFlow) -> None:
        finals = [n for n in flow.nodes if n.is_final]
        assert len(finals) == 1
        assert finals[0].id == "goodbye"

    def test_collect_info_has_criteria(self, flow: ConversationFlow) -> None:
        node = next(n for n in flow.nodes if n.id == "collect_info")
        assert node.completion_criteria is not None
        assert len(node.completion_criteria) == 2
        keys = {c.key for c in node.completion_criteria}
        assert keys == {"patient_name", "preferred_date"}

    def test_collect_info_has_max_turns(self, flow: ConversationFlow) -> None:
        node = next(n for n in flow.nodes if n.id == "collect_info")
        assert node.max_turns == 5

    def test_collect_info_no_skip(self, flow: ConversationFlow) -> None:
        node = next(n for n in flow.nodes if n.id == "collect_info")
        assert node.allow_skip is False

    def test_fallback_edge(self, flow: ConversationFlow) -> None:
        node = next(n for n in flow.nodes if n.id == "collect_info")
        fallback = [e for e in node.edges if e.is_fallback]
        assert len(fallback) == 1
        assert fallback[0].id == "info_to_goodbye"


# ---------------------------------------------------------------------------
# Happy path: greeting → collect_info → confirm → goodbye
# ---------------------------------------------------------------------------


class TestHappyPath:
    """New patient books an appointment successfully."""

    @pytest.mark.anyio
    async def test_full_booking(self, flow: ConversationFlow) -> None:
        adapter = MockAdapter(
            edge_sequence=[
                "greeting_to_collect_info",
                "info_to_confirm",
                "confirm_to_goodbye",
            ]
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)

        r1 = await machine.process_turn("I'm a new patient")
        assert r1.outcome == "transition"
        assert r1.to_node == "collect_info"
        assert machine.current_state == "collect_info"

        r2 = await machine.process_turn("Alice Smith, next Monday please")
        assert r2.outcome == "transition"
        assert r2.to_node == "confirm"

        r3 = await machine.process_turn("No, that's all, thanks!")
        assert r3.outcome == "transition"
        assert r3.to_node == "goodbye"
        assert "Thank you" in r3.response

        assert machine.is_complete
        assert adapter.session_ended
        assert len(machine.context.transition_log) == 3


# ---------------------------------------------------------------------------
# FAQ detour: greeting → faq → collect_info → confirm → goodbye
# ---------------------------------------------------------------------------


class TestFaqDetour:
    """User asks a question first, then books."""

    @pytest.mark.anyio
    async def test_faq_then_book(self, flow: ConversationFlow) -> None:
        adapter = MockAdapter(
            edge_sequence=[
                "greeting_to_faq",
                "faq_to_collect_info",
                "info_to_confirm",
                "confirm_to_goodbye",
            ]
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)

        r1 = await machine.process_turn("What are your hours?")
        assert r1.outcome == "transition"
        assert r1.to_node == "faq"

        r2 = await machine.process_turn("Ok, I'd like to book now")
        assert r2.outcome == "transition"
        assert r2.to_node == "collect_info"

        await machine.process_turn("Bob Jones, Friday")
        await machine.process_turn("Looks good")

        assert machine.current_state == "goodbye"
        assert machine.is_complete
        assert len(machine.context.transition_log) == 4


# ---------------------------------------------------------------------------
# Loop back: confirm → faq → collect_info (re-entry)
# ---------------------------------------------------------------------------


class TestLoopBack:
    """User asks a question after confirmation, then re-enters booking."""

    @pytest.mark.anyio
    async def test_confirm_to_faq_to_collect(self, flow: ConversationFlow) -> None:
        adapter = MockAdapter(
            edge_sequence=[
                "greeting_to_collect_info",
                "info_to_confirm",
                "confirm_to_faq",
                "faq_to_collect_info",
                "info_to_confirm",
                "confirm_to_goodbye",
            ]
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)

        for i in range(6):
            await machine.process_turn(f"turn {i}")

        assert machine.is_complete
        assert len(machine.context.transition_log) == 6

        # collect_info visited twice
        visited = [r.to_node for r in machine.context.transition_log]
        assert visited.count("collect_info") == 2

        # visit_count tracks re-entry
        assert machine.context.visit_count["collect_info"] == 2


# ---------------------------------------------------------------------------
# Early exit: greeting → faq → goodbye
# ---------------------------------------------------------------------------


class TestEarlyExit:
    """User asks a question and leaves without booking."""

    @pytest.mark.anyio
    async def test_faq_to_goodbye(self, flow: ConversationFlow) -> None:
        adapter = MockAdapter(edge_sequence=["greeting_to_faq", "faq_to_goodbye"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("Do you take insurance?")
        await machine.process_turn("Ok thanks, bye")

        assert machine.current_state == "goodbye"
        assert machine.is_complete
        assert len(machine.context.transition_log) == 2


# ---------------------------------------------------------------------------
# Stay turn: criteria not met
# ---------------------------------------------------------------------------


class TestStayTurn:
    """User doesn't provide required info — stays in collect_info."""

    @pytest.mark.anyio
    async def test_stays_when_criteria_not_met(self, flow: ConversationFlow) -> None:
        # First turn transitions to collect_info, then criteria not met
        adapter = MockAdapter(edge_sequence=["greeting_to_collect_info"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        r1 = await machine.process_turn("New patient")
        assert r1.outcome == "transition"
        assert r1.to_node == "collect_info"

        # Second turn: adapter sequence exhausted → all_required_met=False
        r2 = await machine.process_turn("Hmm, not sure about dates")
        assert r2.outcome == "stay"
        assert r2.to_node == "collect_info"
        assert r2.response != ""  # always has a response


# ---------------------------------------------------------------------------
# Strict node blocks skip
# ---------------------------------------------------------------------------


class TestStrictNode:
    """collect_info has allow_skip=False — insistence should not skip."""

    @pytest.mark.anyio
    async def test_insistence_blocked(self, flow: ConversationFlow) -> None:
        adapter = MockAdapterWithCriteria(
            edge_id="info_to_confirm",
            criteria_met={"patient_name": False, "preferred_date": False},
            user_insisting=True,
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)
        # Move to collect_info first
        machine.state = "collect_info"
        machine.context.current_node_id = "collect_info"

        result = await machine.process_turn("Just skip it!")
        assert result.outcome == "stay"
        assert machine.current_state == "collect_info"


# ---------------------------------------------------------------------------
# Visit count and turns_in_node tracking
# ---------------------------------------------------------------------------


class TestTracking:
    """Verify visit_count and turns_in_node behave correctly."""

    @pytest.mark.anyio
    async def test_turns_in_node_increments(self, flow: ConversationFlow) -> None:
        adapter = MockAdapter(edge_sequence=["greeting_to_collect_info"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("New patient")
        assert machine.context.turns_in_node == 0  # reset after transition

        # Stay turns in collect_info
        adapter2 = MockAdapter(edge_sequence=[])
        machine._adapter = adapter2

        await machine.process_turn("hmm")
        assert machine.context.turns_in_node == 1

        await machine.process_turn("still thinking")
        assert machine.context.turns_in_node == 2

    @pytest.mark.anyio
    async def test_visit_count_initial(self, flow: ConversationFlow) -> None:
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # Initial node visited once on construction
        assert machine.context.visit_count["greeting"] == 1
