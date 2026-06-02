"""Tests for CriteriaJudge LLM evaluation prompt builder."""

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

from superdialog.flow.models import CompletionCriterion, Edge, FlowNode  # noqa: E402
from superdialog.machine.criteria import CriteriaJudge  # noqa: E402


def _make_node(
    node_id: str = "greeting",
    edges: list[Edge] | None = None,
    criteria: list[CompletionCriterion] | None = None,
) -> FlowNode:
    return FlowNode(
        id=node_id,
        name=node_id.replace("_", " ").title(),
        instruction="Greet the user",
        edges=edges
        or [
            Edge(
                id="greeting_yes",
                condition="User responds positively",
                target_node_id="next",
            ),
            Edge(
                id="greeting_no",
                condition="User declines",
                target_node_id="end",
            ),
        ],
        completion_criteria=criteria,
    )


class TestCriteriaJudgeBuildPrompt:
    def test_builds_system_prompt_with_edges(self) -> None:
        node = _make_node()
        judge = CriteriaJudge()
        messages = judge.build_evaluation_messages(
            node=node,
            history=[{"role": "user", "content": "Hello!"}],
            userdata={},
            system_prompt="You are a helpful assistant.",
        )
        system_msg = messages[0]["content"]
        assert "greeting_yes" in system_msg
        assert "greeting_no" in system_msg
        assert "User responds positively" in system_msg

    def test_includes_completion_criteria(self) -> None:
        node = _make_node(
            criteria=[
                CompletionCriterion(
                    key="name",
                    description="Collected user name",
                    required=True,
                ),
            ]
        )
        judge = CriteriaJudge()
        messages = judge.build_evaluation_messages(
            node=node,
            history=[{"role": "user", "content": "I'm John"}],
            userdata={},
            system_prompt="",
        )
        system_msg = messages[0]["content"]
        assert "name" in system_msg
        assert "Collected user name" in system_msg

    def test_includes_userdata_context(self) -> None:
        node = _make_node()
        judge = CriteriaJudge()
        messages = judge.build_evaluation_messages(
            node=node,
            history=[],
            userdata={"phone": "1234567890"},
            system_prompt="",
        )
        system_msg = messages[0]["content"]
        assert "1234567890" in system_msg


class TestCriteriaJudgeParseResponse:
    def test_parses_valid_json(self) -> None:
        judge = CriteriaJudge()
        raw = json.dumps(
            {
                "criteria_met": {"name": True},
                "all_required_met": True,
                "user_insisting": False,
                "recommended_edge_id": "greeting_yes",
                "reason": "User said hello",
            }
        )
        result = judge.parse_response("greeting", raw)
        assert result.recommended_edge_id == "greeting_yes"
        assert result.all_required_met is True
        assert result.node_id == "greeting"

    def test_handles_malformed_json_raises(self) -> None:
        judge = CriteriaJudge()
        with pytest.raises(Exception):
            judge.parse_response("greeting", "not json at all")

    def test_extracts_json_from_markdown_code_block(self) -> None:
        judge = CriteriaJudge()
        raw = (
            '```json\n{"recommended_edge_id": "go_next", "all_required_met": true}\n```'
        )
        result = judge.parse_response("n1", raw)
        assert result.recommended_edge_id == "go_next"


class TestCriteriaJudgeEvaluate:
    @pytest.mark.anyio
    async def test_evaluate_calls_llm_and_returns_result(
        self,
    ) -> None:
        async def mock_llm(messages: list[dict]) -> str:  # type: ignore[type-arg]
            return json.dumps(
                {
                    "criteria_met": {},
                    "all_required_met": True,
                    "recommended_edge_id": "greeting_yes",
                    "reason": "positive response",
                }
            )

        judge = CriteriaJudge(llm_fn=mock_llm)
        node = _make_node()
        result = await judge.evaluate(
            node=node,
            history=[{"role": "user", "content": "Hi!"}],
            userdata={},
            system_prompt="",
        )
        assert result.recommended_edge_id == "greeting_yes"

    @pytest.mark.anyio
    async def test_evaluate_raises_on_llm_error(self) -> None:
        async def failing_llm(messages: list[dict]) -> str:  # type: ignore[type-arg]
            raise RuntimeError("LLM down")

        judge = CriteriaJudge(llm_fn=failing_llm)
        node = _make_node()
        with pytest.raises(RuntimeError, match="LLM down"):
            await judge.evaluate(
                node=node,
                history=[],
                userdata={},
                system_prompt="",
            )

    @pytest.mark.anyio
    async def test_evaluate_without_llm_fn(self) -> None:
        judge = CriteriaJudge()
        node = _make_node()
        result = await judge.evaluate(
            node=node,
            history=[],
            userdata={},
            system_prompt="",
        )
        assert result.recommended_edge_id is None
        assert "No LLM" in result.reason
