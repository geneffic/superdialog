"""Tests for run_flow_from_node — direct state injection.

Validates that the fast-forward bug fix works correctly:
the machine starts at the specified node without sending
any [ff:edge_id] messages to the LLM.
"""

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

from superdialog.flow.models import ConversationFlow, Edge, FlowNode  # noqa: E402
from superdialog.machine.runner import run_flow_from_node  # noqa: E402


def _three_node_flow() -> ConversationFlow:
    """a -> b -> c (final)."""
    return ConversationFlow(
        system_prompt="Test.",
        initial_node="a",
        nodes=[
            FlowNode(
                id="a",
                name="A",
                instruction="Ask for name.",
                edges=[
                    Edge(
                        id="got_name",
                        condition="User provides name",
                        target_node_id="b",
                    ),
                ],
            ),
            FlowNode(
                id="b",
                name="B",
                instruction="Confirm name and proceed.",
                edges=[
                    Edge(
                        id="confirmed",
                        condition="User confirms",
                        target_node_id="c",
                    ),
                ],
            ),
            FlowNode(
                id="c",
                name="C",
                static_text="Done!",
                is_final=True,
            ),
        ],
    )


def _make_edge_llm(edge_id: str):
    """LLM that always returns the specified edge."""

    async def llm(messages: list[dict]) -> str:
        sys_content = messages[0].get("content", "")
        if "evaluating" in sys_content:
            return json.dumps(
                {
                    "all_required_met": True,
                    "recommended_edge_id": edge_id,
                    "reason": "mock",
                }
            )
        return "Mock reply"

    return llm


class TestRunFlowFromNode:
    @pytest.mark.anyio
    async def test_start_at_initial_node(self) -> None:
        """Starting at initial node should work like run_flow."""
        flow = _three_node_flow()
        llm = _make_edge_llm("got_name")

        result = await run_flow_from_node(
            flow=flow,
            llm_fn=llm,
            start_node="a",
            user_messages=["I'm Alice"],
        )
        assert len(result.transitions) == 1
        assert result.transitions[0].edge_id == "got_name"
        assert result.transitions[0].to_node == "b"

    @pytest.mark.anyio
    async def test_start_at_middle_node(self) -> None:
        """Starting at node 'b' should skip 'a' entirely."""
        flow = _three_node_flow()
        llm = _make_edge_llm("confirmed")

        result = await run_flow_from_node(
            flow=flow,
            llm_fn=llm,
            start_node="b",
            user_messages=["Yes, confirmed"],
        )
        # Should have transitioned from b -> c
        assert len(result.transitions) == 1
        assert result.transitions[0].from_node == "b"
        assert result.transitions[0].to_node == "c"
        assert result.is_complete is True

    @pytest.mark.anyio
    async def test_no_ff_messages_sent_to_llm(self) -> None:
        """Verify no [ff:...] messages are sent to the LLM."""
        received_messages: list[list[dict]] = []

        async def tracking_llm(messages: list[dict]) -> str:
            received_messages.append(messages)
            sys_content = messages[0].get("content", "")
            if "evaluating" in sys_content:
                return json.dumps(
                    {
                        "all_required_met": True,
                        "recommended_edge_id": "confirmed",
                        "reason": "mock",
                    }
                )
            return "Mock reply"

        flow = _three_node_flow()
        await run_flow_from_node(
            flow=flow,
            llm_fn=tracking_llm,
            start_node="b",
            user_messages=["Yes"],
        )

        # Check that no message contains [ff:
        for msg_list in received_messages:
            for msg in msg_list:
                content = msg.get("content", "")
                assert "[ff:" not in content, f"Found [ff: in LLM message: {content}"

    @pytest.mark.anyio
    async def test_invalid_node_raises(self) -> None:
        """Starting at a non-existent node should raise ValueError."""
        flow = _three_node_flow()
        llm = _make_edge_llm("confirmed")

        with pytest.raises(ValueError, match="not found"):
            await run_flow_from_node(
                flow=flow,
                llm_fn=llm,
                start_node="nonexistent",
                user_messages=["test"],
            )

    @pytest.mark.anyio
    async def test_visit_count_initialized(self) -> None:
        """Visit count should be set for the start node."""
        flow = _three_node_flow()

        async def spy_llm(messages: list[dict]) -> str:
            return json.dumps(
                {
                    "all_required_met": True,
                    "recommended_edge_id": "confirmed",
                    "reason": "mock",
                }
            )

        result = await run_flow_from_node(
            flow=flow,
            llm_fn=spy_llm,
            start_node="b",
            user_messages=["Yes"],
        )
        # Machine should have completed (b -> c)
        assert result.is_complete is True
