"""Tests for dialog-machine-robustness features.

Covers: apply_transition, slot extraction, intent stack, global edges,
agent runtime properties, ToolDescriptor, and get_enriched_instructions.
"""

from __future__ import annotations

import sys
from typing import Any
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
from superdialog.machine.models import CriteriaResult  # noqa: E402
from superdialog.machine.testing.mock_adapter import MockAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Flow helpers
# ---------------------------------------------------------------------------


def _flow_with_global_edges() -> ConversationFlow:
    """Flow: greeting -> collect_info -> confirm -> goodbye.

    Global edge: global_faq -> faq (dead-end, triggers auto-return).
    confirm is non-interruptible.
    """
    return ConversationFlow(
        system_prompt="test",
        initial_node="greeting",
        agent_language="en",
        agent_gender="female",
        global_edges=[
            Edge(
                id="global_faq",
                condition="User asks a general question",
                target_node_id="faq",
            ),
        ],
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                static_text="Hello!",
                interruptible=True,
                edges=[
                    Edge(
                        id="greeting_to_collect",
                        condition="user ready",
                        target_node_id="collect_info",
                    ),
                ],
            ),
            FlowNode(
                id="collect_info",
                name="Collect Info",
                instruction="Collect name and date.",
                interruptible=True,
                completion_criteria=[
                    {"key": "name", "description": "Name", "required": True},
                    {"key": "date", "description": "Date", "required": True},
                ],
                edges=[
                    Edge(
                        id="collect_to_confirm",
                        condition="all collected",
                        target_node_id="confirm",
                    ),
                ],
            ),
            FlowNode(
                id="confirm",
                name="Confirm",
                static_text="Confirmed!",
                interruptible=False,
                edges=[
                    Edge(
                        id="confirm_to_goodbye",
                        condition="confirmed",
                        target_node_id="goodbye",
                    ),
                ],
            ),
            FlowNode(
                id="faq",
                name="FAQ",
                instruction="Answer question.",
                edges=[],  # dead-end → auto-return
            ),
            FlowNode(
                id="goodbye",
                name="Goodbye",
                static_text="Bye!",
                is_final=True,
            ),
        ],
    )


def _simple_flow() -> ConversationFlow:
    """Minimal 3-node flow for apply_transition tests."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                static_text="Hi",
                edges=[
                    Edge(
                        id="e_start_mid",
                        condition="go",
                        target_node_id="middle",
                    ),
                ],
            ),
            FlowNode(
                id="middle",
                name="Middle",
                instruction="Do stuff",
                edges=[
                    Edge(
                        id="e_mid_end",
                        condition="done",
                        target_node_id="end",
                    ),
                ],
            ),
            FlowNode(
                id="end",
                name="End",
                static_text="Bye",
                is_final=True,
            ),
        ],
    )


class _SlotAdapter(MockAdapter):
    """Mock adapter that returns extracted_slots in CriteriaResult."""

    def __init__(
        self,
        edge_sequence: list[str],
        slot_sequence: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(edge_sequence)
        self._slot_sequence = slot_sequence or []
        self._slot_index = 0

    async def evaluate_criteria(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
    ) -> CriteriaResult:
        result = await super().evaluate_criteria(node, history, userdata)
        if self._slot_index < len(self._slot_sequence):
            result.extracted_slots = self._slot_sequence[self._slot_index]
            self._slot_index += 1
        return result


# ---------------------------------------------------------------------------
# 9.2 apply_transition tests
# ---------------------------------------------------------------------------


class TestApplyTransition:
    @pytest.mark.anyio
    async def test_normal_transition(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        result = await machine.apply_transition("e_start_mid")

        assert result.outcome == "transition"
        assert result.from_node == "start"
        assert result.to_node == "middle"
        assert machine.current_state == "middle"

    @pytest.mark.anyio
    async def test_with_user_input(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.apply_transition("e_start_mid", user_input="I'm ready")

        history = machine.context.conversation_history
        user_msgs = [m for m in history if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "I'm ready"

    @pytest.mark.anyio
    async def test_with_collected_data(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.apply_transition(
            "e_start_mid",
            collected_data={"name": "Alice"},
        )

        assert machine.context.node_slots["start"]["name"] == "Alice"
        assert machine.context.userdata["name"] == "Alice"

    @pytest.mark.anyio
    async def test_invalid_edge_raises(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        with pytest.raises(ValueError, match="not valid from state"):
            await machine.apply_transition("e_mid_end")

    @pytest.mark.anyio
    async def test_final_node_noop(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.apply_transition("e_start_mid")
        await machine.apply_transition("e_mid_end")
        assert machine.is_complete

        result = await machine.apply_transition("e_mid_end")
        assert result.outcome == "stay"


# ---------------------------------------------------------------------------
# 9.3 Slot extraction tests
# ---------------------------------------------------------------------------


class TestSlotExtraction:
    @pytest.mark.anyio
    async def test_extracted_slots_merged(self) -> None:
        flow = _flow_with_global_edges()
        adapter = _SlotAdapter(
            edge_sequence=["greeting_to_collect"],
            slot_sequence=[{}],  # greeting turn: no slots
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)
        await machine.process_turn("go")
        assert machine.current_state == "collect_info"

        # Now at collect_info — return slots
        adapter._edge_sequence.append("collect_to_confirm")
        adapter._slot_sequence.append({"name": "Alice", "date": "Monday"})
        await machine.process_turn("Alice, Monday please")

        assert machine.context.node_slots.get("collect_info", {}).get("name") == "Alice"
        assert machine.context.userdata.get("name") == "Alice"

    @pytest.mark.anyio
    async def test_slots_accumulate_across_turns(self) -> None:
        flow = _flow_with_global_edges()
        adapter = _SlotAdapter(
            edge_sequence=[],
            slot_sequence=[{"name": "Bob"}, {"date": "Tuesday"}],
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)
        # Manually set to collect_info
        machine.state = "collect_info"
        machine.context.current_node_id = "collect_info"

        await machine.process_turn("My name is Bob")
        assert machine.context.node_slots["collect_info"]["name"] == "Bob"

        await machine.process_turn("Tuesday works")
        assert machine.context.node_slots["collect_info"]["date"] == "Tuesday"
        assert machine.context.node_slots["collect_info"]["name"] == "Bob"

    @pytest.mark.anyio
    async def test_slot_correction(self) -> None:
        flow = _flow_with_global_edges()
        adapter = _SlotAdapter(
            edge_sequence=[],
            slot_sequence=[{"name": "Bob"}, {"name": "Robert"}],
        )
        machine = await DialogStateMachine.from_flow(flow, adapter)
        machine.state = "collect_info"
        machine.context.current_node_id = "collect_info"

        await machine.process_turn("Bob")
        assert machine.context.node_slots["collect_info"]["name"] == "Bob"

        await machine.process_turn("Actually, Robert")
        assert machine.context.node_slots["collect_info"]["name"] == "Robert"


# ---------------------------------------------------------------------------
# 9.4 Intent stack tests
# ---------------------------------------------------------------------------


class TestIntentStack:
    @pytest.mark.anyio
    async def test_push_on_global_edge(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # At greeting, use apply_transition for global edge
        await machine.apply_transition("global_faq")

        # FAQ is a dead-end → auto-return fires, stack is popped
        # But the push should have happened before the transition
        assert machine.current_state == "greeting"  # auto-returned

    @pytest.mark.anyio
    async def test_auto_return_pops_stack(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # Navigate to collect_info first
        await machine.apply_transition("greeting_to_collect")
        assert machine.current_state == "collect_info"

        # Global edge from collect_info → faq (dead-end → auto-return)
        await machine.apply_transition("global_faq")
        assert machine.current_state == "collect_info"
        assert len(machine.context.intent_stack) == 0

    @pytest.mark.anyio
    async def test_intent_frame_preserves_slots(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.apply_transition("greeting_to_collect")
        # Set some slots in collect_info
        machine.context.node_slots["collect_info"] = {"name": "Alice"}
        machine.context.turns_in_node = 3

        # Global edge → faq → auto-return
        await machine.apply_transition("global_faq")

        assert machine.current_state == "collect_info"
        assert machine.context.node_slots["collect_info"]["name"] == "Alice"
        assert machine.context.turns_in_node == 3

    @pytest.mark.anyio
    async def test_nested_interrupts(self) -> None:
        """Double global edge push — both should pop correctly."""
        # Build a flow with two global edges and a faq that has an edge
        flow = ConversationFlow(
            system_prompt="test",
            initial_node="a",
            global_edges=[
                Edge(id="g1", condition="q1", target_node_id="detour1"),
                Edge(id="g2", condition="q2", target_node_id="detour2"),
            ],
            nodes=[
                FlowNode(
                    id="a",
                    name="A",
                    static_text="A",
                    edges=[
                        Edge(id="a_to_b", condition="go", target_node_id="b"),
                    ],
                ),
                FlowNode(
                    id="b",
                    name="B",
                    static_text="B",
                    is_final=True,
                ),
                FlowNode(
                    id="detour1",
                    name="D1",
                    instruction="d1",
                    edges=[
                        Edge(
                            id="d1_to_d2",
                            condition="deeper",
                            target_node_id="detour2",
                        ),
                    ],
                ),
                FlowNode(
                    id="detour2",
                    name="D2",
                    instruction="d2",
                    edges=[],  # dead-end
                ),
            ],
        )
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # From a → global_faq → detour1
        await machine.apply_transition("g1")
        # detour1 has outgoing edge, so no auto-return
        assert machine.current_state == "detour1"
        assert len(machine.context.intent_stack) == 1

        # From detour1 → g2 → detour2 (dead-end → auto-return to detour1)
        await machine.apply_transition("g2")
        # detour2 is dead-end → auto-return pops to detour1
        assert machine.current_state == "detour1"
        assert len(machine.context.intent_stack) == 1


# ---------------------------------------------------------------------------
# 9.5 Global edges tests
# ---------------------------------------------------------------------------


class TestGlobalEdges:
    @pytest.mark.anyio
    async def test_registered_from_interruptible_nodes(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # global_faq should be a valid trigger from greeting and collect_info
        greeting_triggers = machine._machine.get_triggers("greeting")
        assert "global_faq" in greeting_triggers

        collect_triggers = machine._machine.get_triggers("collect_info")
        assert "global_faq" in collect_triggers

    @pytest.mark.anyio
    async def test_excluded_from_non_interruptible(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # confirm is non-interruptible
        confirm_triggers = machine._machine.get_triggers("confirm")
        assert "global_faq" not in confirm_triggers

    @pytest.mark.anyio
    async def test_excluded_from_final(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        goodbye_triggers = machine._machine.get_triggers("goodbye")
        assert "global_faq" not in goodbye_triggers

    @pytest.mark.anyio
    async def test_is_global_edge(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        assert machine.is_global_edge("global_faq") is True
        assert machine.is_global_edge("greeting_to_collect") is False

    @pytest.mark.anyio
    async def test_global_edge_fires_from_any_interruptible(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # Move to collect_info, then fire global edge
        await machine.apply_transition("greeting_to_collect")
        result = await machine.apply_transition("global_faq")

        # Auto-returned since faq is dead-end
        assert machine.current_state == "collect_info"


# ---------------------------------------------------------------------------
# 9.6 Agent runtime properties tests
# ---------------------------------------------------------------------------


class TestAgentRuntimeProps:
    @pytest.mark.anyio
    async def test_language_init_from_flow(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        assert machine.context.agent_language == "en"

    @pytest.mark.anyio
    async def test_gender_init_from_flow(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        assert machine.context.agent_gender == "female"

    @pytest.mark.anyio
    async def test_set_language_dynamic(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        machine.set_language("hi")
        assert machine.context.agent_language == "hi"

    @pytest.mark.anyio
    async def test_gender_is_static(self) -> None:
        """No set_gender method exists."""
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        assert not hasattr(machine, "set_gender")

    @pytest.mark.anyio
    async def test_default_empty_values(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        assert machine.context.agent_language == ""
        assert machine.context.agent_gender == ""

    @pytest.mark.anyio
    async def test_language_in_flow_meta(self) -> None:
        """Verify _flow_meta passes language/gender to adapter."""
        flow = _flow_with_global_edges()
        captured_meta: dict[str, Any] = {}

        class _CapturingAdapter(MockAdapter):
            async def evaluate_criteria(
                self,
                node: FlowNode,
                history: list[dict[str, Any]],
                userdata: dict[str, Any],
            ) -> CriteriaResult:
                captured_meta.update(userdata.get("_flow_meta", {}))
                return await super().evaluate_criteria(node, history, userdata)

        adapter = _CapturingAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)
        await machine.process_turn("hello")

        assert captured_meta.get("agent_language") == "en"
        assert captured_meta.get("agent_gender") == "female"


# ---------------------------------------------------------------------------
# 9.7 ToolDescriptor tests
# ---------------------------------------------------------------------------


class TestToolDescriptor:
    @pytest.mark.anyio
    async def test_normal_edges(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        tools = machine.get_tools_for_node()
        assert len(tools) == 1
        assert tools[0].id == "e_start_mid"
        assert tools[0].is_global is False
        assert tools[0].is_data_collection is False

    @pytest.mark.anyio
    async def test_data_collection_edge(self) -> None:
        flow = ConversationFlow(
            system_prompt="test",
            initial_node="a",
            nodes=[
                FlowNode(
                    id="a",
                    name="A",
                    instruction="collect",
                    edges=[
                        Edge(
                            id="collect_name",
                            condition="name given",
                            target_node_id="b",
                            input_schema={
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                },
                            },
                        ),
                    ],
                ),
                FlowNode(id="b", name="B", static_text="done", is_final=True),
            ],
        )
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        tools = machine.get_tools_for_node()
        assert len(tools) == 1
        assert tools[0].is_data_collection is True
        assert tools[0].input_schema is not None

    @pytest.mark.anyio
    async def test_global_edges_included(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        tools = machine.get_tools_for_node()
        global_tools = [t for t in tools if t.is_global]
        assert len(global_tools) == 1
        assert global_tools[0].id == "global_faq"

    @pytest.mark.anyio
    async def test_non_interruptible_excludes_global(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        confirm_node = machine._node_map["confirm"]
        tools = machine.get_tools_for_node(confirm_node)
        global_tools = [t for t in tools if t.is_global]
        assert len(global_tools) == 0

    @pytest.mark.anyio
    async def test_final_node_excludes_global(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        goodbye_node = machine._node_map["goodbye"]
        tools = machine.get_tools_for_node(goodbye_node)
        global_tools = [t for t in tools if t.is_global]
        assert len(global_tools) == 0


# ---------------------------------------------------------------------------
# 9.8 get_enriched_instructions tests
# ---------------------------------------------------------------------------


class TestEnrichedInstructions:
    @pytest.mark.anyio
    async def test_includes_instruction(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # Move to collect_info (has instruction)
        await machine.apply_transition("greeting_to_collect")
        text = machine.get_enriched_instructions()
        assert "Collect name and date" in text

    @pytest.mark.anyio
    async def test_includes_slots(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.apply_transition("greeting_to_collect")
        machine.context.node_slots["collect_info"] = {"name": "Alice"}
        text = machine.get_enriched_instructions()
        assert "Alice" in text

    @pytest.mark.anyio
    async def test_includes_visit_count(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # Visit collect_info twice
        await machine.apply_transition("greeting_to_collect")
        machine.context.visit_count["collect_info"] = 3
        text = machine.get_enriched_instructions()
        assert "visited 3 times" in text

    @pytest.mark.anyio
    async def test_includes_language(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        text = machine.get_enriched_instructions()
        assert "Speak in English" in text

    @pytest.mark.anyio
    async def test_includes_gender(self) -> None:
        flow = _flow_with_global_edges()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        text = machine.get_enriched_instructions()
        assert "gender is: female" in text

    @pytest.mark.anyio
    async def test_empty_language_omitted(self) -> None:
        flow = _simple_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        text = machine.get_enriched_instructions()
        assert "Agent language" not in text
        assert "Agent gender" not in text
