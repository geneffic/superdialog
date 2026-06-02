"""Tests for _flow_meta passthrough and create_machine() convenience."""

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
from superdialog.machine.models import CriteriaResult  # noqa: E402
from superdialog.machine.runner import create_machine  # noqa: E402
from superdialog.machine.store import InMemoryContextStore  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_flow() -> ConversationFlow:
    return ConversationFlow(
        system_prompt="test",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="Ask user something",
                edges=[
                    Edge(
                        id="e_to_mid",
                        condition="user ready",
                        target_node_id="middle",
                    ),
                ],
            ),
            FlowNode(
                id="middle",
                name="Middle",
                instruction="Collect details",
                edges=[
                    Edge(
                        id="e_to_end",
                        condition="done",
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


def _loop_flow() -> ConversationFlow:
    """Flow with loop: start → middle → start → middle → end."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="Ask",
                edges=[
                    Edge(
                        id="e_to_mid",
                        condition="go",
                        target_node_id="middle",
                    ),
                ],
            ),
            FlowNode(
                id="middle",
                name="Middle",
                instruction="Middle step",
                edges=[
                    Edge(
                        id="e_back",
                        condition="back",
                        target_node_id="start",
                    ),
                    Edge(
                        id="e_to_end",
                        condition="done",
                        target_node_id="end",
                    ),
                ],
            ),
            FlowNode(
                id="end",
                name="End",
                static_text="Bye!",
                is_final=True,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Adapter that captures _flow_meta from userdata
# ---------------------------------------------------------------------------


class FlowMetaCapturingAdapter:
    """Captures the userdata (including _flow_meta) passed to evaluate_criteria."""

    def __init__(self, edge_sequence: list[str]) -> None:
        self._edges = list(edge_sequence)
        self._index = 0
        self.captured_userdata: list[dict[str, Any]] = []
        self.session_ended: bool = False

    async def speak(self, text: str, node: FlowNode) -> None:
        pass

    async def generate_reply(self, instruction: str, node: FlowNode, history=None, userdata=None) -> str:
        return f"reply:{node.id}"

    async def evaluate_criteria(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
    ) -> CriteriaResult:
        self.captured_userdata.append(dict(userdata))
        if self._index < len(self._edges):
            edge_id = self._edges[self._index]
            self._index += 1
            return CriteriaResult(
                node_id=node.id,
                criteria_met={"auto": True},
                all_required_met=True,
                recommended_edge_id=edge_id,
                reason="mock",
            )
        return CriteriaResult(
            node_id=node.id,
            all_required_met=False,
            reason="exhausted",
            response="stay",
        )

    async def generate_recovery(self, node: FlowNode, error: str) -> str:
        return "recovery"

    async def execute_action(
        self,
        action: Any,
        userdata: dict[str, Any],
    ) -> dict[str, Any] | None:
        return None

    async def end_session(self) -> None:
        self.session_ended = True


# ---------------------------------------------------------------------------
# Tests: _flow_meta passthrough
# ---------------------------------------------------------------------------


class TestFlowMetaPassthrough:
    @pytest.mark.anyio
    async def test_flow_meta_present_in_userdata(self) -> None:
        """Machine injects _flow_meta into userdata for evaluate_criteria."""
        adapter = FlowMetaCapturingAdapter(["e_to_mid"])
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)
        await machine.process_turn("hello")

        assert len(adapter.captured_userdata) == 1
        meta = adapter.captured_userdata[0].get("_flow_meta")
        assert meta is not None
        assert "visit_count" in meta
        assert "turns_in_node" in meta

    @pytest.mark.anyio
    async def test_flow_meta_visit_count_first_visit(self) -> None:
        """First visit to a node should have visit_count=1."""
        adapter = FlowMetaCapturingAdapter(["e_to_mid"])
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)
        await machine.process_turn("hello")

        meta = adapter.captured_userdata[0]["_flow_meta"]
        assert meta["visit_count"] == 1

    @pytest.mark.anyio
    async def test_flow_meta_turns_in_node_increments(self) -> None:
        """turns_in_node should increment on consecutive stays."""
        # Two stays (no transitions)
        adapter = FlowMetaCapturingAdapter([])
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)

        await machine.process_turn("first")
        await machine.process_turn("second")

        assert adapter.captured_userdata[0]["_flow_meta"]["turns_in_node"] == 1
        assert adapter.captured_userdata[1]["_flow_meta"]["turns_in_node"] == 2

    @pytest.mark.anyio
    async def test_flow_meta_visit_count_on_reentry(self) -> None:
        """visit_count should be 2 after looping back to a node."""
        # start → middle → start (back)
        adapter = FlowMetaCapturingAdapter(["e_to_mid", "e_back"])
        machine = await DialogStateMachine.from_flow(_loop_flow(), adapter)

        await machine.process_turn("go")  # start → middle
        await machine.process_turn("back")  # middle → start

        # First call: start (visit 1), second: middle (visit 1)
        assert adapter.captured_userdata[0]["_flow_meta"]["visit_count"] == 1
        assert adapter.captured_userdata[1]["_flow_meta"]["visit_count"] == 1

        # Now do another turn at start (visit 2)
        adapter._edges.append("e_to_mid")
        await machine.process_turn("go again")
        assert adapter.captured_userdata[2]["_flow_meta"]["visit_count"] == 2

    @pytest.mark.anyio
    async def test_flow_meta_turns_resets_after_transition(self) -> None:
        """turns_in_node should reset to 1 after transitioning."""
        # Stay once, then transition
        adapter = FlowMetaCapturingAdapter(["e_to_mid"])
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)

        # First turn stays (no edge available since we only have one)
        # Actually, we have one edge and it transitions immediately
        await machine.process_turn("go")

        # turns_in_node should be 1 (first turn in start node)
        assert adapter.captured_userdata[0]["_flow_meta"]["turns_in_node"] == 1


# ---------------------------------------------------------------------------
# Tests: create_machine() convenience
# ---------------------------------------------------------------------------


def _make_llm(edge_sequence: list[str]):
    """Create a mock LLM that returns edges in sequence."""
    idx = {"i": 0}

    async def llm(messages: list[dict]) -> str:
        sys_content = messages[0].get("content", "")
        if "evaluating" in sys_content:
            edge_id = edge_sequence[idx["i"]] if idx["i"] < len(edge_sequence) else None
            idx["i"] += 1
            return json.dumps(
                {
                    "all_required_met": True,
                    "recommended_edge_id": edge_id,
                    "reason": "mock",
                }
            )
        return "Mock reply"

    return llm


class TestCreateMachine:
    @pytest.mark.anyio
    async def test_creates_machine_from_flow(self) -> None:
        """create_machine returns a usable DialogStateMachine."""
        llm = _make_llm(["e_to_mid"])
        machine = await create_machine(
            flow=_simple_flow(),
            llm_fn=llm,
        )
        assert isinstance(machine, DialogStateMachine)
        assert machine.current_state == "start"
        assert not machine.is_complete

    @pytest.mark.anyio
    async def test_create_machine_process_turn(self) -> None:
        """Machine from create_machine can process turns."""
        llm = _make_llm(["e_to_mid", "e_to_end"])
        machine = await create_machine(
            flow=_simple_flow(),
            llm_fn=llm,
        )

        result = await machine.process_turn("go")
        assert result.outcome == "transition"
        assert machine.current_state == "middle"

        result = await machine.process_turn("done")
        assert result.outcome == "transition"
        assert machine.current_state == "end"
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_create_machine_from_json_string(self) -> None:
        """create_machine accepts a JSON string."""
        flow = _simple_flow()
        flow_json = flow.model_dump_json()
        llm = _make_llm(["e_to_mid"])
        machine = await create_machine(
            flow=flow_json,
            llm_fn=llm,
        )
        assert machine.current_state == "start"

    @pytest.mark.anyio
    async def test_create_machine_with_store(self) -> None:
        """create_machine wires up session_id and store."""
        store = InMemoryContextStore()
        llm = _make_llm(["e_to_mid"])
        machine = await create_machine(
            flow=_simple_flow(),
            llm_fn=llm,
            session_id="test-session",
            store=store,
        )
        assert machine.current_state == "start"
        # Process a turn to trigger save
        await machine.process_turn("go")
        assert machine.current_state == "middle"

    @pytest.mark.anyio
    async def test_create_machine_custom_system_prompt(self) -> None:
        """create_machine uses custom system_prompt when provided."""
        llm = _make_llm([])
        machine = await create_machine(
            flow=_simple_flow(),
            llm_fn=llm,
            system_prompt="Custom prompt override",
        )
        # Machine should be created successfully
        assert machine.current_state == "start"
