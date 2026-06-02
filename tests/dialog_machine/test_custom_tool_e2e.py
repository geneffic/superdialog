"""End-to-end tests for custom tool execution through the DialogStateMachine.

Tests the full lifecycle:
  Flow JSON (with tools) → DialogStateMachine.from_flow(tool_handlers=...)
    → get_tools_for_node() returns custom + edge descriptors
      → execute_tool() invokes handler → data merges into node_slots/userdata
        → get_enriched_instructions() reflects tool results
          → ToolResult.transition_edge_id triggers apply_transition()
            → LiveKit bridge converts custom ToolDescriptors to function_tools

Also tests interaction between custom tools and existing features:
  - Custom tools + global edges + intent stack
  - Custom tools + slot extraction across turns
  - Custom tools on non-interruptible / final nodes
  - Multiple tools fired in sequence within one node
  - Tool data surviving transitions (userdata persists, node_slots reset)
"""

from __future__ import annotations

import json
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
from superdialog.machine.models import ToolDefinition, ToolResult  # noqa: E402
from superdialog.machine.testing.mock_adapter import MockAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Flow definitions
# ---------------------------------------------------------------------------


def _e2e_flow() -> ConversationFlow:
    """Realistic flow with custom tools at both scopes.

    Scenario: appointment booking with inventory check and KB search.

    Flow:
      greeting → collect_info → confirm → goodbye
                                   ↑
      (global) faq ───────────────┘

    Tools:
      flow-level: search_kb (always available, like get_docs)
      node-level on collect_info: check_availability (checks date slots)
      node-level on collect_info: verify_insurance (verifies + optionally transitions)
    """
    return ConversationFlow(
        system_prompt="You are an appointment assistant.",
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
        tools=[
            ToolDefinition(
                id="search_kb",
                name="Search Knowledge Base",
                description="Search documentation for answers",
                handler_id="search_kb",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        ],
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                static_text="Hello! I can help you book an appointment.",
                edges=[
                    Edge(
                        id="e_greet_to_collect",
                        condition="User ready to book",
                        target_node_id="collect_info",
                    ),
                ],
            ),
            FlowNode(
                id="collect_info",
                name="Collect Info",
                instruction="Collect patient name and date.",
                interruptible=True,
                tools=[
                    ToolDefinition(
                        id="check_availability",
                        name="Check Availability",
                        description="Check if a date has available slots",
                        handler_id="check_availability",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "date": {"type": "string"},
                            },
                        },
                    ),
                    ToolDefinition(
                        id="verify_insurance",
                        name="Verify Insurance",
                        description="Verify patient insurance coverage",
                        handler_id="verify_insurance",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "policy_number": {"type": "string"},
                            },
                        },
                    ),
                ],
                edges=[
                    Edge(
                        id="e_collect_to_confirm",
                        condition="All info collected",
                        target_node_id="confirm",
                    ),
                ],
            ),
            FlowNode(
                id="confirm",
                name="Confirm",
                static_text="Great, your appointment is confirmed!",
                interruptible=False,
                edges=[
                    Edge(
                        id="e_confirm_to_bye",
                        condition="User done",
                        target_node_id="goodbye",
                    ),
                ],
            ),
            FlowNode(
                id="faq",
                name="FAQ",
                instruction="Answer the question.",
                edges=[],
            ),
            FlowNode(
                id="goodbye",
                name="Goodbye",
                static_text="Goodbye!",
                is_final=True,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Reusable handler factories
# ---------------------------------------------------------------------------


def _make_handlers() -> dict[str, Any]:
    """Create a set of tool handlers for the E2E flow."""
    call_log: list[dict] = []

    async def search_kb(tool_id: str, args: dict) -> dict:
        call_log.append({"tool": tool_id, "args": args})
        query = args.get("query", "")
        return {
            "kb_results": [f"Doc about {query}"],
            "kb_query": query,
        }

    async def check_availability(tool_id: str, args: dict) -> dict:
        call_log.append({"tool": tool_id, "args": args})
        date = args.get("date", "")
        available = date != "2025-12-25"  # Christmas is full
        return {
            "date_checked": date,
            "slots_available": available,
            "available_times": ["9am", "2pm"] if available else [],
        }

    async def verify_insurance(tool_id: str, args: dict) -> ToolResult:
        call_log.append({"tool": tool_id, "args": args})
        policy = args.get("policy_number", "")
        if policy.startswith("VALID"):
            return ToolResult(
                data={"insurance_verified": True, "policy": policy},
                transition_edge_id="e_collect_to_confirm",
            )
        return ToolResult(
            data={"insurance_verified": False, "policy": policy},
        )

    handlers = {
        "search_kb": search_kb,
        "check_availability": check_availability,
        "verify_insurance": verify_insurance,
    }
    return handlers, call_log


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------


class TestE2EToolLifecycle:
    """Full lifecycle: flow construction → tool execution → data merge."""

    @pytest.mark.anyio
    async def test_data_only_tool_merges_and_stays(self) -> None:
        """check_availability returns dict → data merged, no transition."""
        flow = _e2e_flow()
        handlers, call_log = _make_handlers()
        adapter = MockAdapter(edge_sequence=["e_greet_to_collect"])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        # Move to collect_info
        await machine.process_turn("I want to book")
        assert machine.current_state == "collect_info"

        # Execute check_availability
        result = await machine.execute_tool(
            "check_availability", {"date": "2025-03-20"}
        )

        assert result.outcome == "stay"
        assert machine.current_state == "collect_info"

        # Data merged into node_slots
        slots = machine.context.node_slots["collect_info"]
        assert slots["slots_available"] is True
        assert slots["available_times"] == ["9am", "2pm"]

        # Data merged into userdata
        assert machine.context.userdata["date_checked"] == "2025-03-20"

        # Handler was actually called
        assert len(call_log) == 1
        assert call_log[0]["tool"] == "check_availability"

    @pytest.mark.anyio
    async def test_tool_with_transition_moves_state(self) -> None:
        """verify_insurance returns ToolResult with edge → transition fires."""
        flow = _e2e_flow()
        handlers, call_log = _make_handlers()
        adapter = MockAdapter(edge_sequence=["e_greet_to_collect"])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        await machine.process_turn("book please")
        assert machine.current_state == "collect_info"

        # verify_insurance with valid policy triggers transition
        result = await machine.execute_tool(
            "verify_insurance", {"policy_number": "VALID-123"}
        )

        assert result.outcome == "transition"
        assert machine.current_state == "confirm"
        assert machine.context.userdata["insurance_verified"] is True
        assert (
            len(machine.context.transition_log) == 2
        )  # greet→collect + collect→confirm

    @pytest.mark.anyio
    async def test_tool_without_transition_stays(self) -> None:
        """verify_insurance with invalid policy → data-only, stays."""
        flow = _e2e_flow()
        handlers, call_log = _make_handlers()
        adapter = MockAdapter(edge_sequence=["e_greet_to_collect"])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        await machine.process_turn("book")
        result = await machine.execute_tool(
            "verify_insurance", {"policy_number": "INVALID-999"}
        )

        assert result.outcome == "stay"
        assert machine.current_state == "collect_info"
        assert machine.context.userdata["insurance_verified"] is False

    @pytest.mark.anyio
    async def test_tool_data_visible_in_enriched_instructions(self) -> None:
        """After tool execution, data appears in get_enriched_instructions."""
        flow = _e2e_flow()
        handlers, _ = _make_handlers()
        adapter = MockAdapter(edge_sequence=["e_greet_to_collect"])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        await machine.process_turn("book")
        await machine.execute_tool("check_availability", {"date": "2025-03-20"})

        instructions = machine.get_enriched_instructions()
        assert "slots_available" in instructions
        assert "2025-03-20" in instructions


class TestE2EMultipleToolCalls:
    """Multiple tool calls within one node, data accumulates."""

    @pytest.mark.anyio
    async def test_sequential_tools_accumulate_data(self) -> None:
        """Call search_kb then check_availability — both results in slots."""
        flow = _e2e_flow()
        handlers, call_log = _make_handlers()
        adapter = MockAdapter(edge_sequence=["e_greet_to_collect"])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        await machine.process_turn("book")

        # First tool: search_kb
        await machine.execute_tool("search_kb", {"query": "insurance"})
        # Second tool: check_availability
        await machine.execute_tool("check_availability", {"date": "2025-04-01"})

        slots = machine.context.node_slots["collect_info"]
        # Both tool results present
        assert "kb_results" in slots
        assert "slots_available" in slots
        assert slots["kb_query"] == "insurance"
        assert slots["date_checked"] == "2025-04-01"

        # Both handlers called
        assert len(call_log) == 2

    @pytest.mark.anyio
    async def test_tool_then_edge_transition(self) -> None:
        """Use a data-only tool, then transition via process_turn."""
        flow = _e2e_flow()
        handlers, _ = _make_handlers()
        adapter = MockAdapter(
            edge_sequence=["e_greet_to_collect", "e_collect_to_confirm"]
        )
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        # greeting → collect_info
        await machine.process_turn("book")
        assert machine.current_state == "collect_info"

        # Use tool (data-only)
        await machine.execute_tool("check_availability", {"date": "2025-05-01"})
        assert machine.current_state == "collect_info"

        # CriteriaJudge picks the edge → collect_info → confirm
        await machine.process_turn("Alice, next Monday")
        assert machine.current_state == "confirm"

        # Tool data from collect_info still in userdata
        assert machine.context.userdata["date_checked"] == "2025-05-01"


class TestE2EFlowToolScoping:
    """Flow-scoped tools available everywhere, node-scoped only on their node."""

    @pytest.mark.anyio
    async def test_flow_tool_available_on_every_node(self) -> None:
        """search_kb (flow-level) available on greeting, collect, confirm, goodbye."""
        flow = _e2e_flow()
        handlers, _ = _make_handlers()
        adapter = MockAdapter(
            edge_sequence=[
                "e_greet_to_collect",
                "e_collect_to_confirm",
                "e_confirm_to_bye",
            ]
        )
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        for expected_state in ["greeting", "collect_info", "confirm", "goodbye"]:
            descriptors = machine.get_tools_for_node()
            tool_ids = {d.id for d in descriptors}
            assert (
                "search_kb" in tool_ids
            ), f"search_kb missing on {machine.current_state}"

            if expected_state != "goodbye":
                await machine.process_turn("next")

    @pytest.mark.anyio
    async def test_node_tool_not_on_other_nodes(self) -> None:
        """check_availability only on collect_info, not on greeting."""
        flow = _e2e_flow()
        handlers, _ = _make_handlers()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        # On greeting
        descriptors = machine.get_tools_for_node()
        tool_ids = {d.id for d in descriptors}
        assert "check_availability" not in tool_ids
        assert "verify_insurance" not in tool_ids

    @pytest.mark.anyio
    async def test_flow_tool_on_non_interruptible_node(self) -> None:
        """search_kb available on confirm (interruptible=False)."""
        flow = _e2e_flow()
        handlers, _ = _make_handlers()
        adapter = MockAdapter(
            edge_sequence=["e_greet_to_collect", "e_collect_to_confirm"]
        )
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        await machine.process_turn("go")
        await machine.process_turn("done")
        assert machine.current_state == "confirm"
        assert not machine.current_node.interruptible

        descriptors = machine.get_tools_for_node()
        tool_ids = {d.id for d in descriptors}
        assert "search_kb" in tool_ids
        # But global_faq should NOT be here (non-interruptible)
        assert "global_faq" not in tool_ids

    @pytest.mark.anyio
    async def test_flow_tool_on_final_node(self) -> None:
        """search_kb available on goodbye (is_final=True)."""
        flow = _e2e_flow()
        handlers, _ = _make_handlers()
        adapter = MockAdapter(
            edge_sequence=[
                "e_greet_to_collect",
                "e_collect_to_confirm",
                "e_confirm_to_bye",
            ]
        )
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        await machine.process_turn("go")
        await machine.process_turn("done")
        await machine.process_turn("bye")
        assert machine.is_complete

        descriptors = machine.get_tools_for_node()
        tool_ids = {d.id for d in descriptors}
        assert "search_kb" in tool_ids


class TestE2EToolsWithGlobalEdges:
    """Custom tools interact correctly with global edge interrupts."""

    @pytest.mark.anyio
    async def test_tool_data_survives_global_edge_interrupt(self) -> None:
        """Call tool in collect_info, get interrupted by FAQ, data persists."""
        flow = _e2e_flow()
        handlers, _ = _make_handlers()
        adapter = MockAdapter(
            edge_sequence=[
                "e_greet_to_collect",
                # no more edges — will stay in collect_info
            ]
        )
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        # Move to collect_info
        await machine.process_turn("book")
        assert machine.current_state == "collect_info"

        # Use tool — data goes into collect_info slots
        await machine.execute_tool("check_availability", {"date": "2025-06-15"})
        assert machine.context.node_slots["collect_info"]["slots_available"] is True

        # Global edge interrupt: collect_info → faq
        await machine.apply_transition("global_faq")
        assert machine.current_state == "faq"

        # Intent stack was pushed
        assert len(machine.context.intent_stack) == 1
        frame = machine.context.intent_stack[0]
        assert frame.node_id == "collect_info"

        # Tool data still in userdata (persists across nodes)
        assert machine.context.userdata["date_checked"] == "2025-06-15"

    @pytest.mark.anyio
    async def test_flow_tool_usable_during_faq_detour(self) -> None:
        """search_kb callable while on the FAQ detour node."""
        flow = _e2e_flow()
        handlers, call_log = _make_handlers()
        adapter = MockAdapter(edge_sequence=["e_greet_to_collect"])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        await machine.process_turn("book")
        await machine.apply_transition("global_faq")
        assert machine.current_state == "faq"

        # search_kb is flow-level → still available
        result = await machine.execute_tool("search_kb", {"query": "hours"})
        assert result.outcome == "stay"
        assert machine.context.userdata["kb_query"] == "hours"


class TestE2EToolDescriptorFlags:
    """Verify ToolDescriptor metadata is correct for bridge layer consumption."""

    @pytest.mark.anyio
    async def test_descriptor_flags_on_collect_node(self) -> None:
        """collect_info has edge tools + node tools + flow tools + global edges."""
        flow = _e2e_flow()
        handlers, _ = _make_handlers()
        adapter = MockAdapter(edge_sequence=["e_greet_to_collect"])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        await machine.process_turn("go")
        descriptors = machine.get_tools_for_node()

        by_id = {d.id: d for d in descriptors}

        # Edge tool
        assert "e_collect_to_confirm" in by_id
        edge_desc = by_id["e_collect_to_confirm"]
        assert edge_desc.is_custom is False
        assert edge_desc.is_global is False
        assert edge_desc.handler_id is None

        # Global edge
        assert "global_faq" in by_id
        faq_desc = by_id["global_faq"]
        assert faq_desc.is_global is True
        assert faq_desc.is_custom is False

        # Node-scoped custom tool
        assert "check_availability" in by_id
        avail_desc = by_id["check_availability"]
        assert avail_desc.is_custom is True
        assert avail_desc.handler_id == "check_availability"
        assert avail_desc.input_schema is not None

        # Flow-scoped custom tool
        assert "search_kb" in by_id
        kb_desc = by_id["search_kb"]
        assert kb_desc.is_custom is True
        assert kb_desc.handler_id == "search_kb"


class TestE2ELivekitBridgeIntegration:
    """Verify LiveKit bridge handles all descriptor types from E2E flow."""

    @pytest.mark.anyio
    async def test_bridge_generates_tools_for_all_types(self) -> None:
        """descriptors_to_livekit_tools handles edges, globals, and custom."""
        from superdialog.machine.adapters.livekit_bridge import (
            descriptors_to_livekit_tools,
        )

        flow = _e2e_flow()
        handlers, _ = _make_handlers()
        adapter = MockAdapter(edge_sequence=["e_greet_to_collect"])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        await machine.process_turn("go")
        descriptors = machine.get_tools_for_node()

        # Should have: 1 edge + 1 global + 2 node tools + 1 flow tool = 5
        assert len(descriptors) == 5

        tools = descriptors_to_livekit_tools(descriptors, machine)
        assert len(tools) == 5


class TestE2EFullTraversal:
    """Complete flow traversal using tools and transitions together."""

    @pytest.mark.anyio
    async def test_full_happy_path_with_tools(self) -> None:
        """greeting → collect_info (use tools) → confirm → goodbye.

        1. Transition to collect_info via process_turn
        2. Call check_availability (data-only)
        3. Call verify_insurance with valid policy (triggers transition to confirm)
        4. Transition to goodbye via process_turn
        """
        flow = _e2e_flow()
        handlers, call_log = _make_handlers()
        adapter = MockAdapter(
            edge_sequence=[
                "e_greet_to_collect",
                # verify_insurance triggers e_collect_to_confirm
                "e_confirm_to_bye",
            ]
        )
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        # 1. greeting → collect_info
        r1 = await machine.process_turn("I want to book")
        assert r1.outcome == "transition"
        assert machine.current_state == "collect_info"

        # 2. Check availability (data-only)
        r2 = await machine.execute_tool("check_availability", {"date": "2025-07-10"})
        assert r2.outcome == "stay"
        assert machine.context.userdata["slots_available"] is True

        # 3. Verify insurance → triggers transition to confirm
        r3 = await machine.execute_tool(
            "verify_insurance", {"policy_number": "VALID-456"}
        )
        assert r3.outcome == "transition"
        assert machine.current_state == "confirm"
        assert machine.context.userdata["insurance_verified"] is True

        # 4. confirm → goodbye
        r4 = await machine.process_turn("bye")
        assert r4.outcome == "transition"
        assert machine.current_state == "goodbye"
        assert machine.is_complete
        assert adapter.session_ended is True

        # Verify full transition log
        assert len(machine.context.transition_log) == 3
        edges = [t.edge_id for t in machine.context.transition_log]
        assert edges == [
            "e_greet_to_collect",
            "e_collect_to_confirm",
            "e_confirm_to_bye",
        ]

        # Verify all tool handlers were called
        assert len(call_log) == 2
        assert call_log[0]["tool"] == "check_availability"
        assert call_log[1]["tool"] == "verify_insurance"

        # Verify all tool data persisted in userdata
        ud = machine.context.userdata
        assert ud["date_checked"] == "2025-07-10"
        assert ud["slots_available"] is True
        assert ud["insurance_verified"] is True
        assert ud["policy"] == "VALID-456"

    @pytest.mark.anyio
    async def test_full_path_with_faq_detour_and_tools(self) -> None:
        """greeting → collect_info → (tool) → FAQ detour → auto-return → confirm → bye.

        1. Move to collect_info
        2. Use check_availability
        3. Global FAQ interrupt (intent stack push)
        4. Use search_kb during FAQ
        5. Auto-return to collect_info (dead-end FAQ)
        6. verify_insurance triggers transition to confirm
        7. process_turn to goodbye
        """
        flow = _e2e_flow()
        handlers, call_log = _make_handlers()
        adapter = MockAdapter(
            edge_sequence=[
                "e_greet_to_collect",
                # FAQ is a dead-end — auto-return after
                "e_confirm_to_bye",
            ]
        )
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers=handlers
        )

        # 1. greeting → collect_info
        await machine.process_turn("book")
        assert machine.current_state == "collect_info"

        # 2. Check availability
        await machine.execute_tool("check_availability", {"date": "2025-08-01"})

        # 3. FAQ interrupt
        await machine.apply_transition("global_faq")
        assert machine.current_state == "faq"

        # 4. Use search_kb during FAQ
        await machine.execute_tool("search_kb", {"query": "parking"})
        assert machine.context.userdata["kb_query"] == "parking"

        # 5. FAQ is dead-end → auto-return to collect_info happened
        # (auto-return fires inside _do_transition when the detour node is dead-end)
        assert machine.current_state == "collect_info"

        # Original tool data survived the detour
        assert machine.context.userdata["date_checked"] == "2025-08-01"

        # 6. verify_insurance triggers transition
        r = await machine.execute_tool(
            "verify_insurance", {"policy_number": "VALID-789"}
        )
        assert r.outcome == "transition"
        assert machine.current_state == "confirm"

        # 7. confirm → goodbye
        await machine.process_turn("done")
        assert machine.is_complete

        # All 3 handlers called
        assert len(call_log) == 3

    @pytest.mark.anyio
    async def test_error_handler_missing(self) -> None:
        """execute_tool with unregistered handler raises ValueError."""
        flow = _e2e_flow()

        # Register only search_kb — leave check_availability unregistered
        async def search_kb(tool_id: str, args: dict) -> dict:
            return {}

        adapter = MockAdapter(edge_sequence=["e_greet_to_collect"])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers={"search_kb": search_kb}
        )

        await machine.process_turn("book")

        with pytest.raises(ValueError, match="No handler registered"):
            await machine.execute_tool("check_availability", {"date": "x"})


class TestE2EFromJSON:
    """Test custom tools declared via JSON string parsing."""

    @pytest.mark.anyio
    async def test_flow_from_json_with_tools(self) -> None:
        """Parse a flow JSON string with tools and verify E2E."""
        flow_json = json.dumps(
            {
                "system_prompt": "Test",
                "initial_node": "start",
                "tools": [
                    {
                        "id": "platform_search",
                        "name": "Search",
                        "description": "Search docs",
                        "handler_id": "platform_search",
                    }
                ],
                "nodes": [
                    {
                        "id": "start",
                        "name": "Start",
                        "instruction": "Welcome",
                        "tools": [
                            {
                                "id": "local_lookup",
                                "name": "Lookup",
                                "description": "Local lookup",
                                "handler_id": "local_lookup",
                                "input_schema": {
                                    "type": "object",
                                    "properties": {
                                        "key": {"type": "string"},
                                    },
                                },
                            }
                        ],
                        "edges": [
                            {
                                "id": "e_go",
                                "condition": "done",
                                "target_node_id": "end",
                            }
                        ],
                    },
                    {
                        "id": "end",
                        "name": "End",
                        "static_text": "Bye",
                        "is_final": True,
                    },
                ],
            }
        )

        flow = ConversationFlow.from_json_string(flow_json)

        # Verify parsing
        assert len(flow.tools) == 1
        assert flow.tools[0].id == "platform_search"
        start_node = next(n for n in flow.nodes if n.id == "start")
        assert len(start_node.tools) == 1
        assert start_node.tools[0].id == "local_lookup"
        assert start_node.tools[0].input_schema is not None

        # Wire up handlers and run
        async def platform_search(tool_id: str, args: dict) -> dict:
            return {"found": True}

        async def local_lookup(tool_id: str, args: dict) -> dict:
            return {"value": args.get("key", ""), "resolved": True}

        adapter = MockAdapter(edge_sequence=["e_go"])
        machine = await DialogStateMachine.from_flow(
            flow,
            adapter,
            tool_handlers={
                "platform_search": platform_search,
                "local_lookup": local_lookup,
            },
        )

        # Use both tools
        r1 = await machine.execute_tool("platform_search", {})
        assert r1.outcome == "stay"
        assert machine.context.userdata["found"] is True

        r2 = await machine.execute_tool("local_lookup", {"key": "test"})
        assert r2.outcome == "stay"
        assert machine.context.userdata["resolved"] is True

        # Transition via CriteriaJudge
        r3 = await machine.process_turn("done")
        assert machine.is_complete

        # All data persisted
        assert machine.context.userdata["found"] is True
        assert machine.context.userdata["resolved"] is True
