"""Tests for DialogStateMachine."""

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

from superdialog.flow.models import ConversationFlow, Edge, FlowNode  # noqa: E402
from superdialog.machine.machine import DialogStateMachine  # noqa: E402
from superdialog.machine.testing.mock_adapter import MockAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KAIRALI_FLOW_PATH = Path(__file__).resolve().parents[4] / "temp" / "kairali_flow 2.json"


def _simple_flow() -> ConversationFlow:
    """Build a simple 3-node flow: start -> middle -> end."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                static_text="Hello!",
                edges=[
                    Edge(
                        id="e_start_mid",
                        condition="user ready",
                        target_node_id="middle",
                    ),
                ],
            ),
            FlowNode(
                id="middle",
                name="Middle",
                instruction="Ask user something",
                edges=[
                    Edge(
                        id="e_mid_end",
                        condition="user done",
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


def _load_kairali_flow() -> ConversationFlow:
    """Load kairali flow from JSON file."""
    return ConversationFlow.from_json_file(str(KAIRALI_FLOW_PATH))


# ---------------------------------------------------------------------------
# TestMachineConstruction
# ---------------------------------------------------------------------------


class TestMachineConstruction:
    """Tests for building a DialogStateMachine from a ConversationFlow."""

    @pytest.mark.anyio
    async def test_initial_state(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        assert machine.current_state == "start"
        assert machine.current_node.id == "start"

    @pytest.mark.anyio
    async def test_is_complete_false_initially(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        assert machine.is_complete is False

    @pytest.mark.anyio
    async def test_has_trigger(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        assert machine.has_trigger("e_start_mid") is True
        assert machine.has_trigger("e_mid_end") is True
        assert machine.has_trigger("nonexistent_edge") is False

    @pytest.mark.anyio
    async def test_current_node_returns_flow_node(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        node = machine.current_node
        assert node.name == "Start"
        assert node.static_text == "Hello!"

    @pytest.mark.anyio
    async def test_context_initialized(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        assert machine.context.current_node_id == "start"
        assert machine.context.conversation_history == []
        assert machine.context.transition_log == []

    @pytest.mark.skipif(
        not KAIRALI_FLOW_PATH.exists(),
        reason="kairali_flow 2.json not found",
    )
    @pytest.mark.anyio
    async def test_kairali_flow_construction(self) -> None:
        flow = _load_kairali_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        assert machine.current_state == "greeting"
        assert machine.is_complete is False
        assert machine.has_trigger("greeting_yes") is True
        assert machine.has_trigger("greeting_wrong_number") is True
        assert machine.has_trigger("contact_confirm_to_close") is True

    @pytest.mark.skipif(
        not KAIRALI_FLOW_PATH.exists(),
        reason="kairali_flow 2.json not found",
    )
    @pytest.mark.anyio
    async def test_kairali_final_states(self) -> None:
        flow = _load_kairali_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # The flow has several final states
        final_ids = machine._final_states
        assert "closing" in final_ids
        assert "unqualified_close" in final_ids
        assert "already_spoke_close" in final_ids
        assert "nobody_enquired_close" in final_ids
        assert "greeting" not in final_ids


# ---------------------------------------------------------------------------
# TestProcessTurn
# ---------------------------------------------------------------------------


class TestProcessTurn:
    """Tests for process_turn behavior."""

    @pytest.mark.anyio
    async def test_single_transition(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=["e_start_mid"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("I'm ready")

        assert machine.current_state == "middle"
        assert machine.is_complete is False
        assert len(machine.context.transition_log) == 1
        assert machine.context.transition_log[0].from_node == "start"
        assert machine.context.transition_log[0].to_node == "middle"
        assert machine.context.transition_log[0].edge_id == "e_start_mid"

    @pytest.mark.anyio
    async def test_full_traversal(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=["e_start_mid", "e_mid_end"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("go")
        assert machine.current_state == "middle"
        # middle has instruction, so generate_reply should be called
        assert len(adapter.replies) == 1

        await machine.process_turn("done")
        assert machine.current_state == "end"
        assert machine.is_complete is True
        # end has static_text, so speak should be called
        assert "Goodbye!" in adapter.spoken
        # session should be ended at final node
        assert adapter.session_ended is True

    @pytest.mark.anyio
    async def test_no_transition_when_no_edge_recommended(self) -> None:
        flow = _simple_flow()
        # Empty sequence means evaluate_criteria returns no edge
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("hello")

        assert machine.current_state == "start"
        assert len(machine.context.transition_log) == 0

    @pytest.mark.anyio
    async def test_conversation_history_tracked(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("first message")
        await machine.process_turn("second message")

        history = machine.context.conversation_history
        # History now includes both user and assistant messages
        user_msgs = [m for m in history if m["role"] == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[0] == {"role": "user", "content": "first message"}
        assert user_msgs[1] == {"role": "user", "content": "second message"}

    @pytest.mark.anyio
    async def test_no_action_after_complete(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=["e_start_mid", "e_mid_end", "extra"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("go")
        await machine.process_turn("done")
        assert machine.is_complete is True

        # Further turns should be no-ops
        await machine.process_turn("more input")
        assert len(machine.context.transition_log) == 2
        # History includes user + assistant messages from 2 turns
        # (4 total). Third turn is a no-op — no new messages added.
        user_msgs = [
            m for m in machine.context.conversation_history if m["role"] == "user"
        ]
        assert len(user_msgs) == 2

    @pytest.mark.anyio
    async def test_invalid_edge_from_current_state(self) -> None:
        """Edge recommended but not valid from current state."""
        flow = _simple_flow()
        # e_mid_end is not valid from "start"
        adapter = MockAdapter(edge_sequence=["e_mid_end"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("try wrong edge")

        assert machine.current_state == "start"
        assert len(machine.context.transition_log) == 0

    @pytest.mark.anyio
    async def test_transition_with_speak(self) -> None:
        """Transitioning to a node with static_text calls speak."""
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=["e_start_mid"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # middle has instruction (not static_text), so generate_reply
        await machine.process_turn("go")
        assert len(adapter.replies) == 1
        assert len(adapter.spoken) == 0

    @pytest.mark.anyio
    async def test_skipped_flag_in_transition_record(self) -> None:
        """When all_required_met is True, skipped should be False."""
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=["e_start_mid"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("go")
        record = machine.context.transition_log[0]
        assert record.skipped is False


# ---------------------------------------------------------------------------
# TestKairaliTraversal
# ---------------------------------------------------------------------------


class TestKairaliTraversal:
    """Tests for traversing the kairali flow."""

    @pytest.mark.skipif(
        not KAIRALI_FLOW_PATH.exists(),
        reason="kairali_flow 2.json not found",
    )
    @pytest.mark.anyio
    async def test_wrong_number_fast_exit(self) -> None:
        """greeting -> unqualified_close (1 edge)."""
        flow = _load_kairali_flow()
        adapter = MockAdapter(edge_sequence=["greeting_wrong_number"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("wrong number")

        assert machine.current_state == "unqualified_close"
        assert machine.is_complete is True
        assert adapter.session_ended is True
        assert len(machine.context.transition_log) == 1

    @pytest.mark.skipif(
        not KAIRALI_FLOW_PATH.exists(),
        reason="kairali_flow 2.json not found",
    )
    @pytest.mark.anyio
    async def test_products_self_use_full_path(self) -> None:
        """Full path: greeting -> availability -> enquiry_open ->
        products_self_or_business -> products_self_use_capture ->
        products_self_use_callback -> contact_capture ->
        contact_capture_email -> contact_capture_city ->
        contact_capture_time -> contact_capture_confirm -> closing.

        That is 11 edges total.
        """
        flow = _load_kairali_flow()
        edge_sequence = [
            "greeting_yes",
            "available_yes",
            "enquiry_products",
            "products_self_use",
            "products_self_use_captured",
            "products_self_to_contact",
            "alt_number_given",
            "email_given",
            "city_given",
            "callback_time_given",
            "contact_confirm_to_close",
        ]
        adapter = MockAdapter(edge_sequence=edge_sequence)
        machine = await DialogStateMachine.from_flow(flow, adapter)

        for i in range(len(edge_sequence)):
            await machine.process_turn(f"turn {i}")

        assert machine.current_state == "closing"
        assert machine.is_complete is True
        assert adapter.session_ended is True
        assert len(machine.context.transition_log) == 11

    @pytest.mark.skipif(
        not KAIRALI_FLOW_PATH.exists(),
        reason="kairali_flow 2.json not found",
    )
    @pytest.mark.anyio
    async def test_general_info_loop_back(self) -> None:
        """Test loop: greeting -> availability -> enquiry_open ->
        kairali_general_info -> enquiry_open (loop back) ->
        then pick a different category and close.

        enquiry_open is visited twice.
        """
        flow = _load_kairali_flow()
        edge_sequence = [
            "greeting_yes",
            "available_yes",
            "enquiry_general_kairali",
            "general_info_wants_more",  # loops back to enquiry_open
            "enquiry_careers",
            "careers_no_callback",  # goes to closing
        ]
        adapter = MockAdapter(edge_sequence=edge_sequence)
        machine = await DialogStateMachine.from_flow(flow, adapter)

        for i in range(len(edge_sequence)):
            await machine.process_turn(f"turn {i}")

        assert machine.current_state == "closing"
        assert machine.is_complete is True
        assert adapter.session_ended is True
        assert len(machine.context.transition_log) == 6

        # Verify enquiry_open was visited twice
        visited = [r.to_node for r in machine.context.transition_log]
        assert visited.count("enquiry_open") == 2

    @pytest.mark.skipif(
        not KAIRALI_FLOW_PATH.exists(),
        reason="kairali_flow 2.json not found",
    )
    @pytest.mark.anyio
    async def test_healing_village_treatment_path(self) -> None:
        """greeting -> availability -> enquiry_open ->
        healing_village_type -> healing_village_treatment_capture ->
        healing_village_booking_type -> healing_village_callback ->
        contact_capture -> contact_capture_email ->
        contact_capture_city -> contact_capture_time ->
        contact_capture_confirm -> closing.
        """
        flow = _load_kairali_flow()
        edge_sequence = [
            "greeting_yes",
            "available_yes",
            "enquiry_healing_village",
            "hv_treatment",
            "hv_condition_captured",
            "hv_booking_type_captured",
            "hv_to_contact",
            "alt_number_skipped",
            "email_skipped",
            "city_given",
            "callback_time_given",
            "contact_confirm_to_close",
        ]
        adapter = MockAdapter(edge_sequence=edge_sequence)
        machine = await DialogStateMachine.from_flow(flow, adapter)

        for i in range(len(edge_sequence)):
            await machine.process_turn(f"turn {i}")

        assert machine.current_state == "closing"
        assert machine.is_complete is True
        assert adapter.session_ended is True
        assert len(machine.context.transition_log) == 12
