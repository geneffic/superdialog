"""Tests for @tool decorator and plain-function tool registration."""

import sys
from typing import Any
from unittest.mock import MagicMock

# Stub livekit modules so tests run without the full livekit install
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
    ConversationFlow,
    Edge,
    FlowNode,
    ToolDefinition,
)
from superdialog.machine.machine import DialogStateMachine  # noqa: E402
from superdialog.machine.testing.mock_adapter import MockAdapter  # noqa: E402
from superdialog.tools import PythonTool, tool  # noqa: E402

# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


class TestToolDecorator:
    def test_bare_decorator_returns_python_tool(self) -> None:
        @tool
        async def lookup_customer(customer_id: str) -> dict:
            """Look up a customer."""
            return {"id": customer_id}

        assert isinstance(lookup_customer, PythonTool)
        assert lookup_customer.id == "lookup_customer"
        assert lookup_customer.description == "Look up a customer."

    def test_bare_decorator_infers_schema(self) -> None:
        @tool
        async def check_inventory(sku: str, qty: int) -> dict:
            """Check stock."""
            return {}

        schema = lookup_tool_schema(check_inventory)
        assert schema["properties"]["sku"]["type"] == "string"
        assert schema["properties"]["qty"]["type"] == "integer"
        assert "sku" in schema["required"]

    def test_decorator_with_name_override(self) -> None:
        @tool(name="check_stock")
        async def check_inventory(sku: str) -> dict:
            """Check stock."""
            return {}

        assert check_inventory.id == "check_stock"

    def test_decorator_with_description_override(self) -> None:
        @tool(description="Custom description")
        async def my_fn(x: str) -> dict:
            return {}

        assert my_fn.description == "Custom description"

    def test_function_tool_factory_form(self) -> None:
        """tool(fn) without @ syntax also works."""

        async def lookup(customer_id: str) -> dict:
            """Lookup."""
            return {}

        t = tool(lookup)
        assert isinstance(t, PythonTool)
        assert t.id == "lookup"

    @pytest.mark.anyio
    async def test_decorated_tool_executes(self) -> None:
        @tool
        async def greet(name: str) -> dict:
            """Greet someone."""
            return {"greeting": f"Hello, {name}"}

        result = await greet.execute({"name": "Alice"})
        assert result.data == {"greeting": "Hello, Alice"}


# ---------------------------------------------------------------------------
# Plain callables in tools=[] on from_flow
# ---------------------------------------------------------------------------


def _simple_flow() -> ConversationFlow:
    return ConversationFlow(
        system_prompt="Test",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="Begin",
                edges=[Edge(id="e_done", condition="done", target_node_id="end")],
            ),
            FlowNode(id="end", name="End", static_text="Bye", is_final=True),
        ],
    )


class TestPlainCallableInToolsList:
    @pytest.mark.anyio
    async def test_plain_fn_auto_wrapped(self) -> None:
        async def lookup_customer(customer_id: str) -> dict:
            """Look up a customer."""
            return {"found": True, "id": customer_id}

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(
            _simple_flow(), adapter, tools=[lookup_customer]
        )

        result = await machine.execute_tool("lookup_customer", {"customer_id": "abc"})
        assert result.outcome == "stay"
        assert machine.context.userdata["found"] is True

    @pytest.mark.anyio
    async def test_decorated_tool_in_tools_list(self) -> None:
        @tool
        async def check_inventory(sku: str) -> dict:
            """Check inventory."""
            return {"available": True, "sku": sku}

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(
            _simple_flow(), adapter, tools=[check_inventory]
        )

        result = await machine.execute_tool("check_inventory", {"sku": "X1"})
        assert result.outcome == "stay"
        assert machine.context.userdata["available"] is True

    @pytest.mark.anyio
    async def test_mixed_tool_and_callable_in_tools_list(self) -> None:
        """PythonTool instances and plain callables can be mixed."""

        async def plain_fn(x: str) -> dict:
            """Plain."""
            return {"plain": x}

        explicit_tool = PythonTool.of(plain_fn, name="explicit")

        async def another_fn(y: str) -> dict:
            """Another."""
            return {"another": y}

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(
            _simple_flow(), adapter, tools=[explicit_tool, another_fn]
        )

        r1 = await machine.execute_tool("explicit", {"x": "hi"})
        assert r1.outcome == "stay"
        assert machine.context.userdata["plain"] == "hi"

        r2 = await machine.execute_tool("another_fn", {"y": "yo"})
        assert r2.outcome == "stay"
        assert machine.context.userdata["another"] == "yo"


# ---------------------------------------------------------------------------
# Function reference on ConversationFlow.tools and FlowNode.tools
# ---------------------------------------------------------------------------


class TestFnRefOnFlowModel:
    def test_callable_coerced_to_tool_definition_on_flow(self) -> None:
        async def search_kb(query: str) -> dict:
            """Search the knowledge base."""
            return {}

        flow = ConversationFlow(
            system_prompt="Test",
            initial_node="start",
            tools=[search_kb],
            nodes=[FlowNode(id="start", name="Start", static_text="Hi", is_final=True)],
        )

        assert len(flow.tools) == 1
        td = flow.tools[0]
        assert isinstance(td, ToolDefinition)
        assert td.id == "search_kb"
        assert td.description == "Search the knowledge base."
        assert td.handler is search_kb
        assert td.input_schema is not None
        assert "query" in td.input_schema["properties"]

    def test_callable_coerced_to_tool_definition_on_node(self) -> None:
        async def check_availability(date: str) -> dict:
            """Check availability."""
            return {}

        flow = ConversationFlow(
            system_prompt="Test",
            initial_node="start",
            nodes=[
                FlowNode(
                    id="start",
                    name="Start",
                    instruction="Hi",
                    tools=[check_availability],
                    edges=[],
                )
            ],
        )

        td = flow.nodes[0].tools[0]
        assert td.id == "check_availability"
        assert td.handler is check_availability

    @pytest.mark.anyio
    async def test_flow_fn_ref_executes_via_machine(self) -> None:
        async def search_kb(query: str) -> dict:
            """Search KB."""
            return {"results": [f"doc:{query}"]}

        flow = ConversationFlow(
            system_prompt="Test",
            initial_node="start",
            tools=[search_kb],
            nodes=[
                FlowNode(
                    id="start",
                    name="Start",
                    instruction="Hi",
                    edges=[Edge(id="e_end", condition="done", target_node_id="end")],
                ),
                FlowNode(id="end", name="End", static_text="Bye", is_final=True),
            ],
        )

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        result = await machine.execute_tool("search_kb", {"query": "pricing"})
        assert result.outcome == "stay"
        assert machine.context.userdata["results"] == ["doc:pricing"]

    @pytest.mark.anyio
    async def test_node_fn_ref_executes_on_correct_node(self) -> None:
        async def check_availability(date: str) -> dict:
            """Check slots."""
            return {"available": date != "2025-12-25"}

        flow = ConversationFlow(
            system_prompt="Test",
            initial_node="collect",
            nodes=[
                FlowNode(
                    id="collect",
                    name="Collect",
                    instruction="Collect date",
                    tools=[check_availability],
                    edges=[Edge(id="e_done", condition="done", target_node_id="end")],
                ),
                FlowNode(id="end", name="End", static_text="Bye", is_final=True),
            ],
        )

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        result = await machine.execute_tool(
            "check_availability", {"date": "2025-06-01"}
        )
        assert result.outcome == "stay"
        assert machine.context.userdata["available"] is True

    @pytest.mark.anyio
    async def test_decorated_fn_ref_on_flow(self) -> None:
        """@tool decorated function can be passed directly on ConversationFlow.tools."""

        @tool
        async def lookup_customer(customer_id: str) -> dict:
            """Look up customer."""
            return {"id": customer_id, "active": True}

        flow = ConversationFlow(
            system_prompt="Test",
            initial_node="start",
            tools=[lookup_customer],
            nodes=[
                FlowNode(
                    id="start",
                    name="Start",
                    instruction="Hi",
                    edges=[Edge(id="e_end", condition="done", target_node_id="end")],
                ),
                FlowNode(id="end", name="End", static_text="Bye", is_final=True),
            ],
        )

        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        result = await machine.execute_tool("lookup_customer", {"customer_id": "u1"})
        assert result.outcome == "stay"
        assert machine.context.userdata["active"] is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def lookup_tool_schema(t: PythonTool) -> dict[str, Any]:
    assert t.input_schema is not None
    return t.input_schema
