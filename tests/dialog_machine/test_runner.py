import json
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

from superdialog.flow.models import ConversationFlow, Edge, FlowNode  # noqa: E402
from superdialog.machine.runner import FlowResult, run_flow  # noqa: E402


def _simple_flow() -> ConversationFlow:
    """3-node flow: greeting -> collect_name -> goodbye."""
    return ConversationFlow(
        system_prompt="You are a friendly assistant.",
        initial_node="greeting",
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                instruction="Say hello and ask for the user's name.",
                edges=[
                    Edge(
                        id="name_given",
                        condition="User provides name",
                        target_node_id="collect_name",
                    ),
                ],
            ),
            FlowNode(
                id="collect_name",
                name="Collect Name",
                instruction="Confirm the name and say goodbye.",
                edges=[
                    Edge(
                        id="confirmed",
                        condition="User confirms",
                        target_node_id="goodbye",
                    ),
                ],
            ),
            FlowNode(
                id="goodbye",
                name="Goodbye",
                static_text="Thank you! Goodbye.",
                is_final=True,
            ),
        ],
    )


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


class TestRunFlowScripted:
    @pytest.mark.anyio
    async def test_runs_through_simple_flow(self):
        flow = _simple_flow()
        llm = _make_llm(["name_given", "confirmed"])
        result = await run_flow(
            flow=flow,
            llm_fn=llm,
            user_messages=["Hi, I'm Alice", "Yes that's correct"],
        )
        assert isinstance(result, FlowResult)
        assert result.final_state == "goodbye"
        assert result.is_complete is True
        assert len(result.transitions) == 2
        assert result.transitions[0].edge_id == "name_given"
        assert result.transitions[1].edge_id == "confirmed"

    @pytest.mark.anyio
    async def test_returns_partial_when_messages_exhausted(self):
        flow = _simple_flow()
        llm = _make_llm(["name_given"])
        result = await run_flow(
            flow=flow,
            llm_fn=llm,
            user_messages=["Hi, I'm Alice"],
        )
        assert result.final_state == "collect_name"
        assert result.is_complete is False
        assert len(result.transitions) == 1

    @pytest.mark.anyio
    async def test_loads_flow_from_json_string(self):
        flow = _simple_flow()
        flow_json = flow.model_dump_json()
        llm = _make_llm(["name_given", "confirmed"])
        result = await run_flow(
            flow=flow_json,
            llm_fn=llm,
            user_messages=["Hi", "Yes"],
        )
        assert result.is_complete is True


class TestRunFlowFromFile:
    @pytest.mark.anyio
    async def test_loads_from_json_file(self, tmp_path):
        flow = _simple_flow()
        p = tmp_path / "flow.json"
        p.write_text(flow.model_dump_json())
        llm = _make_llm(["name_given", "confirmed"])
        result = await run_flow(
            flow=str(p),
            llm_fn=llm,
            user_messages=["Hi", "Yes"],
        )
        assert result.is_complete is True


class TestFlowResult:
    @pytest.mark.anyio
    async def test_result_has_conversation_history(self):
        flow = _simple_flow()
        llm = _make_llm(["name_given", "confirmed"])
        result = await run_flow(
            flow=flow,
            llm_fn=llm,
            user_messages=["Hi, I'm Alice", "Yes"],
        )
        assert len(result.conversation_history) >= 2
        assert result.conversation_history[0]["role"] == "user"

    @pytest.mark.anyio
    async def test_result_has_responses(self):
        flow = _simple_flow()
        llm = _make_llm(["name_given", "confirmed"])
        result = await run_flow(
            flow=flow,
            llm_fn=llm,
            user_messages=["Hi", "Yes"],
        )
        # goodbye node has static_text "Thank you! Goodbye.", should appear
        assert any("Goodbye" in r for r in result.responses)
