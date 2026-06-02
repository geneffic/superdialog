"""Tests for LiveKit bridge layer.

Uses a minimal fake LiveKit SDK so we can test the bridge logic without
requiring livekit-agents installed.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

# Allow importing dialog_machine and livekit_flows without the LiveKit Agent SDK.
for _mod in [
    "livekit.agents",
    "livekit.agents.llm",
    "livekit.agents.llm.tool_context",
    "livekit.agents.voice",
    "livekit.api",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from superdialog.flow.models import ConversationFlow, Edge, FlowNode
from superdialog.machine.machine import DialogStateMachine
from superdialog.machine.testing.mock_adapter import MockAdapter


def _install_fake_livekit(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install a minimal fake LiveKit SDK into sys.modules for this test."""

    class FakeTool:
        def __init__(
            self,
            fn: Any,
            *,
            name: str | None = None,
            description: str | None = None,
            raw_schema: dict[str, Any] | None = None,
        ) -> None:
            self.fn = fn
            self.name = (
                name
                or (raw_schema or {}).get("name")
                or getattr(fn, "__name__", "tool")
            )
            self.description = description or (raw_schema or {}).get("description", "")
            self.raw_schema = raw_schema

        async def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return await self.fn(*args, **kwargs)

    class FakeRunContext:
        pass

    class FakeSession:
        def __init__(self) -> None:
            self.said: list[str] = []
            self.generated: list[str] = []

        def say(self, text: str) -> None:
            self.said.append(text)

        def generate_reply(self, *, instructions: str) -> None:
            self.generated.append(instructions)

    class FakeAgent:
        def __init__(
            self, *, instructions: str, tools: list[Any], **kwargs: Any
        ) -> None:
            self.instructions = instructions
            self.tools = tools
            self.chat_ctx = kwargs.get("chat_ctx")
            self.session = FakeSession()

    class FakeAgentTask(FakeAgent):
        def __init__(
            self, *, instructions: str, tools: list[Any], **kwargs: Any
        ) -> None:
            super().__init__(instructions=instructions, tools=tools, **kwargs)
            self.completed: Any | None = None

        def complete(self, value: Any) -> None:
            self.completed = value

    def function_tool(
        fn: Any,
        *,
        name: str | None = None,
        description: str | None = None,
        raw_schema: dict[str, Any] | None = None,
    ) -> FakeTool:
        return FakeTool(fn, name=name, description=description, raw_schema=raw_schema)

    livekit = ModuleType("livekit")
    agents = ModuleType("livekit.agents")
    voice = ModuleType("livekit.agents.voice")
    llm = ModuleType("livekit.agents.llm")
    tool_context = ModuleType("livekit.agents.llm.tool_context")
    api = ModuleType("livekit.api")

    agents.Agent = FakeAgent  # type: ignore[attr-defined]
    agents.RunContext = FakeRunContext  # type: ignore[attr-defined]
    agents.function_tool = function_tool  # type: ignore[attr-defined]
    voice.AgentTask = FakeAgentTask  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "livekit", livekit)
    monkeypatch.setitem(sys.modules, "livekit.agents", agents)
    monkeypatch.setitem(sys.modules, "livekit.agents.voice", voice)
    monkeypatch.setitem(sys.modules, "livekit.agents.llm", llm)
    monkeypatch.setitem(sys.modules, "livekit.agents.llm.tool_context", tool_context)
    monkeypatch.setitem(sys.modules, "livekit.api", api)
    return agents


def _bridge_flow() -> ConversationFlow:
    """Flow for bridge tests with local, global, and data-collection edges."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="greeting",
        agent_language="en",
        global_edges=[
            Edge(
                id="global_dead",
                condition="question (dead-end detour)",
                target_node_id="detour_dead",
            ),
            Edge(
                id="global_task",
                condition="question (task detour)",
                target_node_id="detour_task",
            ),
        ],
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                static_text="Hello!",
                edges=[
                    Edge(
                        id="go",
                        condition="user ready",
                        target_node_id="middle",
                    ),
                    Edge(
                        id="collect_name",
                        condition="collect name",
                        target_node_id="middle",
                        input_schema={
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    ),
                ],
            ),
            FlowNode(
                id="middle",
                name="Middle",
                instruction="Do stuff",
                edges=[Edge(id="done", condition="done", target_node_id="end")],
            ),
            FlowNode(id="detour_dead", name="DDead", instruction="dead", edges=[]),
            FlowNode(
                id="detour_task",
                name="DTask",
                instruction="task",
                edges=[
                    Edge(
                        id="detour_done",
                        condition="done detour",
                        target_node_id="detour_task_end",
                    )
                ],
            ),
            FlowNode(
                id="detour_task_end",
                name="DTaskEnd",
                instruction="task end",
                edges=[],  # dead-end triggers auto-return
            ),
            FlowNode(id="end", name="End", static_text="Bye", is_final=True),
        ],
    )


class TestDescriptorsToLivekitTools:
    @pytest.mark.anyio
    async def test_tool_generation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_livekit(monkeypatch)
        from superdialog.machine.adapters.livekit_bridge import (
            descriptors_to_livekit_tools,
        )

        flow = _bridge_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        tools = descriptors_to_livekit_tools(machine.get_tools_for_node(), machine)
        assert {t.name for t in tools} >= {
            "go",
            "collect_name",
            "global_dead",
            "global_task",
        }

    @pytest.mark.anyio
    async def test_data_collection_tool_merges_userdata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agents = _install_fake_livekit(monkeypatch)
        from superdialog.machine.adapters.livekit_bridge import (
            descriptors_to_livekit_tools,
        )

        flow = _bridge_flow()
        adapter = MockAdapter(edge_sequence=[])
        machine = await DialogStateMachine.from_flow(flow, adapter)

        descriptors = [
            d for d in machine.get_tools_for_node() if d.id == "collect_name"
        ]
        tools = descriptors_to_livekit_tools(descriptors, machine)
        assert len(tools) == 1

        await tools[0]({"name": "Alice"}, agents.RunContext())
        assert machine.current_state == "middle"
        assert machine.context.userdata["name"] == "Alice"
        assert machine.context.node_slots["greeting"]["name"] == "Alice"


class TestFlowNodeTask:
    def test_class_creation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_livekit(monkeypatch)
        from superdialog.machine.adapters import livekit_bridge

        importlib.reload(livekit_bridge)
        cls = livekit_bridge.get_flow_node_task_class()
        assert callable(cls)


class TestLazyImport:
    def test_core_import_without_livekit(self) -> None:
        from superdialog.machine.machine import DialogStateMachine
        from superdialog.machine.models import (
            FlowContext,
            IntentFrame,
            ToolDescriptor,
            TurnResult,
        )

        assert DialogStateMachine is not None
        assert FlowContext is not None
        assert IntentFrame is not None
        assert ToolDescriptor is not None
        assert TurnResult is not None

    def test_bridge_module_importable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_livekit(monkeypatch)
        from superdialog.machine.adapters import livekit_bridge

        assert hasattr(livekit_bridge, "descriptors_to_livekit_tools")
        assert hasattr(livekit_bridge, "get_flow_node_task_class")
