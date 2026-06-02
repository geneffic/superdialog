"""Tests for custom tool execution feature.

Covers: ToolDefinition, ToolResult, ToolDescriptor extensions,
flow JSON parsing with tools, execute_tool(), tool availability,
and LiveKit bridge custom tool support.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock LiveKit SDK before any imports
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
from superdialog.machine.models import (  # noqa: E402
    ToolDefinition,
    ToolDescriptor,
    ToolResult,
)
from superdialog.machine.testing.mock_adapter import MockAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flow_with_tools() -> ConversationFlow:
    """Flow with custom tools at flow and node level."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="start",
        agent_language="en",
        tools=[
            ToolDefinition(
                id="search_docs",
                name="Search Docs",
                description="Search documentation",
                handler_id="search_docs",
            ),
        ],
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="Welcome the user",
                edges=[
                    Edge(
                        id="e_go",
                        condition="user ready",
                        target_node_id="middle",
                    ),
                ],
                tools=[
                    ToolDefinition(
                        id="check_inventory",
                        name="Check Inventory",
                        description="Check product stock",
                        handler_id="check_inventory",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "product": {"type": "string"},
                            },
                        },
                    ),
                ],
            ),
            FlowNode(
                id="middle",
                name="Middle",
                instruction="Do stuff",
                interruptible=False,
                edges=[
                    Edge(
                        id="e_done",
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


# ---------------------------------------------------------------------------
# 5. Tests — Models
# ---------------------------------------------------------------------------


class TestToolDefinition:
    def test_construction(self) -> None:
        td = ToolDefinition(
            id="t1",
            name="Tool One",
            description="A tool",
            handler_id="h1",
        )
        assert td.id == "t1"
        assert td.name == "Tool One"
        assert td.handler_id == "h1"
        assert td.input_schema is None

    def test_with_input_schema(self) -> None:
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        td = ToolDefinition(
            id="t2",
            name="Tool Two",
            description="A tool",
            input_schema=schema,
        )
        assert td.input_schema == schema

    def test_serialization(self) -> None:
        td = ToolDefinition(id="t1", name="T", description="D", handler_id="h")
        data = td.model_dump()
        assert data["id"] == "t1"
        assert data["handler_id"] == "h"
        restored = ToolDefinition.model_validate(data)
        assert restored == td


class TestToolResult:
    def test_data_only(self) -> None:
        tr = ToolResult(data={"status": "ok"})
        assert tr.data == {"status": "ok"}
        assert tr.transition_edge_id is None

    def test_with_transition(self) -> None:
        tr = ToolResult(data={"verified": True}, transition_edge_id="e_go")
        assert tr.transition_edge_id == "e_go"

    def test_empty_default(self) -> None:
        tr = ToolResult()
        assert tr.data == {}
        assert tr.transition_edge_id is None


class TestToolDescriptorExtended:
    def test_defaults(self) -> None:
        td = ToolDescriptor(id="e1", description="edge")
        assert td.is_custom is False
        assert td.handler_id is None

    def test_custom_fields(self) -> None:
        td = ToolDescriptor(
            id="t1",
            description="custom",
            is_custom=True,
            handler_id="my_handler",
        )
        assert td.is_custom is True
        assert td.handler_id == "my_handler"


class TestFlowJsonParsing:
    def test_flow_tools_parsed(self) -> None:
        flow = _flow_with_tools()
        assert len(flow.tools) == 1
        assert flow.tools[0].id == "search_docs"

    def test_node_tools_parsed(self) -> None:
        flow = _flow_with_tools()
        start = next(n for n in flow.nodes if n.id == "start")
        assert len(start.tools) == 1
        assert start.tools[0].id == "check_inventory"
        assert start.tools[0].input_schema is not None


# ---------------------------------------------------------------------------
# 6. Tests — Machine Core (execute_tool)
# ---------------------------------------------------------------------------


class TestExecuteTool:
    @pytest.mark.anyio
    async def test_data_only_handler(self) -> None:
        """Dict return merges data, no transition."""
        flow = _flow_with_tools()

        async def handler(tool_id: str, args: dict) -> dict:
            return {"available": True, "stock": 42}

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers={"check_inventory": handler}
        )

        result = await machine.execute_tool("check_inventory", {"product": "X"})

        assert result.outcome == "stay"
        assert machine.current_state == "start"
        # Data merged into node_slots
        slots = machine.context.node_slots.get("start", {})
        assert slots["available"] is True
        assert slots["stock"] == 42
        # Data merged into userdata
        assert machine.context.userdata["available"] is True

    @pytest.mark.anyio
    async def test_transitioning_handler(self) -> None:
        """ToolResult with transition_edge_id triggers transition."""
        flow = _flow_with_tools()

        async def handler(tool_id: str, args: dict) -> ToolResult:
            return ToolResult(data={"verified": True}, transition_edge_id="e_go")

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers={"check_inventory": handler}
        )

        result = await machine.execute_tool("check_inventory", {})

        assert result.outcome == "transition"
        assert machine.current_state == "middle"
        assert machine.context.userdata["verified"] is True

    @pytest.mark.anyio
    async def test_data_visible_in_enriched_instructions(self) -> None:
        """Tool result data queryable via get_enriched_instructions()."""
        flow = _flow_with_tools()

        async def handler(tool_id: str, args: dict) -> dict:
            return {"available": True}

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers={"check_inventory": handler}
        )

        await machine.execute_tool("check_inventory", {})
        instructions = machine.get_enriched_instructions()
        assert "available" in instructions

    @pytest.mark.anyio
    async def test_missing_handler_raises(self) -> None:
        """ValueError when handler_id not found."""
        flow = _flow_with_tools()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter, tool_handlers={})

        with pytest.raises(ValueError, match="No handler registered"):
            await machine.execute_tool("check_inventory", {})

    @pytest.mark.anyio
    async def test_tool_handlers_stored(self) -> None:
        """tool_handlers parameter stored on machine."""
        flow = _flow_with_tools()

        async def handler(tool_id: str, args: dict) -> dict:
            return {}

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(
            flow,
            adapter,
            tool_handlers={"check_inventory": handler},
        )
        assert "check_inventory" in machine._tool_handlers


# ---------------------------------------------------------------------------
# 7. Tests — Tool Availability
# ---------------------------------------------------------------------------


class TestToolAvailability:
    @pytest.mark.anyio
    async def test_custom_tools_alongside_edges(self) -> None:
        """get_tools_for_node returns custom + edge tools."""
        flow = _flow_with_tools()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        descriptors = machine.get_tools_for_node()
        ids = {d.id for d in descriptors}
        # Edge tool
        assert "e_go" in ids
        # Node-scoped custom tool
        assert "check_inventory" in ids
        # Flow-scoped custom tool
        assert "search_docs" in ids

    @pytest.mark.anyio
    async def test_flow_tools_on_non_interruptible(self) -> None:
        """Flow tools available even on non-interruptible nodes."""
        flow = _flow_with_tools()
        adapter = MockAdapter(edge_sequence=["e_go"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # Move to middle (non-interruptible)
        await machine.process_turn("go")
        assert machine.current_state == "middle"

        descriptors = machine.get_tools_for_node()
        ids = {d.id for d in descriptors}
        # Flow tool still available
        assert "search_docs" in ids

    @pytest.mark.anyio
    async def test_flow_tools_on_final_node(self) -> None:
        """Flow tools available on final nodes."""
        flow = _flow_with_tools()
        adapter = MockAdapter(edge_sequence=["e_go", "e_done"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        await machine.process_turn("go")
        await machine.process_turn("done")
        assert machine.is_complete

        descriptors = machine.get_tools_for_node()
        ids = {d.id for d in descriptors}
        assert "search_docs" in ids

    @pytest.mark.anyio
    async def test_node_tools_only_on_their_node(self) -> None:
        """Node-scoped tools not available on other nodes."""
        flow = _flow_with_tools()
        adapter = MockAdapter(edge_sequence=["e_go"])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        # Move to middle — check_inventory is start-only
        await machine.process_turn("go")
        assert machine.current_state == "middle"

        descriptors = machine.get_tools_for_node()
        ids = {d.id for d in descriptors}
        assert "check_inventory" not in ids

    @pytest.mark.anyio
    async def test_custom_descriptor_flags(self) -> None:
        """Custom tool descriptors have is_custom=True and handler_id."""
        flow = _flow_with_tools()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        descriptors = machine.get_tools_for_node()
        inv = next(d for d in descriptors if d.id == "check_inventory")
        assert inv.is_custom is True
        assert inv.handler_id == "check_inventory"
        assert inv.input_schema is not None

        search = next(d for d in descriptors if d.id == "search_docs")
        assert search.is_custom is True
        assert search.handler_id == "search_docs"


# ---------------------------------------------------------------------------
# 8. Tests — LiveKit Bridge Custom Tools
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="superdialog.machine.adapters.livekit_bridge not ported; see Task 7 follow-up"
)
class TestLivekitBridgeCustomTools:
    @pytest.mark.anyio
    async def test_custom_descriptor_to_livekit_tool(self) -> None:
        """Custom tool descriptors converted to LiveKit function_tools."""
        from superdialog.machine.adapters.livekit_bridge import (
            descriptors_to_livekit_tools,
        )

        flow = _flow_with_tools()

        async def handler(tool_id: str, args: dict) -> dict:
            return {}

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers={"check_inventory": handler}
        )

        descriptors = [d for d in machine.get_tools_for_node() if d.is_custom]
        assert len(descriptors) >= 1

        tools = descriptors_to_livekit_tools(descriptors, machine)
        assert len(tools) == len(descriptors)

    @pytest.mark.anyio
    async def test_custom_with_schema_produces_raw_schema(self) -> None:
        """Custom tool with input_schema creates function_tool with raw_schema."""
        from superdialog.machine.adapters.livekit_bridge import (
            descriptors_to_livekit_tools,
        )

        flow = _flow_with_tools()

        async def handler(tool_id: str, args: dict) -> dict:
            return {}

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(
            flow, adapter, tool_handlers={"check_inventory": handler}
        )

        desc = next(
            d for d in machine.get_tools_for_node() if d.id == "check_inventory"
        )
        assert desc.input_schema is not None

        tools = descriptors_to_livekit_tools([desc], machine)
        assert len(tools) == 1

    @pytest.mark.anyio
    async def test_custom_without_schema(self) -> None:
        """Custom tool without input_schema creates parameterless function_tool."""
        from superdialog.machine.adapters.livekit_bridge import (
            descriptors_to_livekit_tools,
        )

        flow = _flow_with_tools()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        desc = next(d for d in machine.get_tools_for_node() if d.id == "search_docs")
        assert desc.input_schema is None

        tools = descriptors_to_livekit_tools([desc], machine)
        assert len(tools) == 1
