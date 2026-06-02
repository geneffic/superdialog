"""Tests for TextAdapter -- provider-agnostic text/chat adapter."""

from __future__ import annotations

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

from superdialog.flow.models import Edge, FlowNode  # noqa: E402
from superdialog.machine.adapters.text_adapter import TextAdapter  # noqa: E402
from superdialog.machine.criteria import CriteriaJudge  # noqa: E402


def _node(node_id: str = "test") -> FlowNode:
    return FlowNode(
        id=node_id,
        name="Test Node",
        instruction="Say hello",
        edges=[
            Edge(
                id="next",
                condition="done",
                target_node_id="end",
            )
        ],
    )


def _make_adapter(
    reply: str = "Hello!",
    edge_id: str = "next",
) -> TextAdapter:
    async def mock_llm(messages: list[dict]) -> str:
        sys_content = messages[0].get("content", "")
        if "evaluating" in sys_content:
            return json.dumps(
                {
                    "all_required_met": True,
                    "recommended_edge_id": edge_id,
                    "reason": "test",
                }
            )
        return reply

    judge = CriteriaJudge(llm_fn=mock_llm)
    return TextAdapter(
        llm_fn=mock_llm,
        criteria_judge=judge,
        system_prompt="Be helpful",
    )


class TestTextAdapterSpeak:
    @pytest.mark.anyio
    async def test_speak_records_text(self) -> None:
        adapter = _make_adapter()
        await adapter.speak("Welcome!", _node())
        assert "Welcome!" in adapter.responses


class TestTextAdapterGenerateReply:
    @pytest.mark.anyio
    async def test_generate_reply_calls_llm(self) -> None:
        adapter = _make_adapter(reply="Hi there!")
        result = await adapter.generate_reply("Greet user", _node())
        assert result == "Hi there!"
        assert "Hi there!" in adapter.responses


class TestTextAdapterEvaluateCriteria:
    @pytest.mark.anyio
    async def test_evaluate_criteria_returns_result(self) -> None:
        adapter = _make_adapter(edge_id="next")
        node = _node()
        result = await adapter.evaluate_criteria(
            node,
            [{"role": "user", "content": "done"}],
            {},
        )
        assert result.recommended_edge_id == "next"
        assert result.all_required_met is True


class TestTextAdapterEndSession:
    @pytest.mark.anyio
    async def test_end_session_marks_complete(self) -> None:
        adapter = _make_adapter()
        assert not adapter.session_ended
        await adapter.end_session()
        assert adapter.session_ended


class TestTextAdapterProtocol:
    def test_implements_runtime_adapter(self) -> None:
        from superdialog.machine.adapters.base import RuntimeAdapter

        adapter = _make_adapter()
        assert isinstance(adapter, RuntimeAdapter)
