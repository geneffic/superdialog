"""Tier 2: Edge accuracy eval tests.

Tests that the LLM picks the correct edge for each utterance
in the test corpus. Uses cached LLM responses for CI speed.

Run with: pytest test_edge_accuracy.py -v --flow=<path> [--no-cache]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
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
from superdialog.machine.eval.cache import ResponseCache  # noqa: E402
from superdialog.machine.eval.evaluator import FlowEvaluator  # noqa: E402
from superdialog.machine.eval.models import EdgeTest, NegativeEdgeResult  # noqa: E402

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _simple_flow() -> ConversationFlow:
    """Simple 3-node flow for testing evaluator mechanics."""
    return ConversationFlow(
        system_prompt="You are a helpful assistant.",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="Greet the user and ask how to help.",
                edges=[
                    Edge(
                        id="wants_help",
                        condition="User asks for help",
                        target_node_id="help",
                    ),
                    Edge(
                        id="wants_bye",
                        condition="User says goodbye",
                        target_node_id="end",
                    ),
                ],
            ),
            FlowNode(
                id="help",
                name="Help",
                instruction="Help the user with their question.",
                edges=[
                    Edge(
                        id="helped",
                        condition="User is satisfied",
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


def _make_deterministic_llm(edge_map: dict[str, str]):
    """Create a mock LLM that picks edges based on keyword matching."""

    async def llm(messages: list[dict]) -> str:
        sys_content = messages[0].get("content", "")
        if "evaluating" in sys_content:
            # Find the user message to determine which edge
            user_msg = ""
            for m in messages:
                if m.get("role") == "user":
                    user_msg = m.get("content", "")

            # Match keywords to edges
            for keyword, edge_id in edge_map.items():
                if keyword.lower() in user_msg.lower():
                    return json.dumps(
                        {
                            "all_required_met": True,
                            "recommended_edge_id": edge_id,
                            "reason": f"matched '{keyword}'",
                        }
                    )
            # Default: return first available edge
            return json.dumps(
                {
                    "all_required_met": False,
                    "recommended_edge_id": None,
                    "reason": "no match",
                }
            )
        return "Mock reply"

    return llm


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestEdgeEvalWithMockLLM:
    """Test evaluator mechanics with deterministic mock LLM."""

    @pytest.mark.anyio
    async def test_correct_edge_passes(self) -> None:
        flow = _simple_flow()
        llm = _make_deterministic_llm({"help": "wants_help", "bye": "wants_bye"})
        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: llm,
        )

        edge_test = EdgeTest(
            node_id="start",
            edge_id="wants_help",
            condition="User asks for help",
            utterances=["I need help please"],
        )
        result = await evaluator.eval_edge(edge_test, "I need help please", "mock")
        assert result.passed is True
        assert result.actual_edge == "wants_help"

    @pytest.mark.anyio
    async def test_wrong_edge_fails(self) -> None:
        flow = _simple_flow()
        llm = _make_deterministic_llm({"help": "wants_help", "bye": "wants_bye"})
        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: llm,
        )

        edge_test = EdgeTest(
            node_id="start",
            edge_id="wants_help",
            condition="User asks for help",
            utterances=["bye bye"],
        )
        result = await evaluator.eval_edge(edge_test, "bye bye", "mock")
        assert result.passed is False
        assert result.actual_edge == "wants_bye"

    @pytest.mark.anyio
    async def test_nonexistent_node_returns_error(self) -> None:
        flow = _simple_flow()
        llm = _make_deterministic_llm({})
        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: llm,
        )

        edge_test = EdgeTest(
            node_id="nonexistent",
            edge_id="some_edge",
            condition="test",
        )
        result = await evaluator.eval_edge(edge_test, "test", "mock")
        assert result.error is not None
        assert "not found" in result.error


class TestNegativeEdgeEval:
    """Test negative utterance evaluation."""

    @pytest.mark.anyio
    async def test_negative_utterance_passes(self) -> None:
        """Negative utterance should NOT trigger the specified edge."""
        flow = _simple_flow()
        llm = _make_deterministic_llm({"help": "wants_help", "bye": "wants_bye"})
        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: llm,
        )

        edge_test = EdgeTest(
            node_id="start",
            edge_id="wants_help",
            condition="User asks for help",
            negative_utterances=["bye bye"],
        )
        result = await evaluator.eval_negative_edge(edge_test, "bye bye", "mock")
        assert isinstance(result, NegativeEdgeResult)
        assert result.passed is True
        assert result.actual_edge == "wants_bye"

    @pytest.mark.anyio
    async def test_negative_utterance_fails(self) -> None:
        """If negative utterance triggers the forbidden edge, it fails."""
        flow = _simple_flow()
        llm = _make_deterministic_llm({"help": "wants_help", "bye": "wants_bye"})
        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: llm,
        )

        edge_test = EdgeTest(
            node_id="start",
            edge_id="wants_help",
            condition="User asks for help",
            negative_utterances=["I need help"],
        )
        result = await evaluator.eval_negative_edge(edge_test, "I need help", "mock")
        assert result.passed is False
        assert result.actual_edge == "wants_help"

    @pytest.mark.anyio
    async def test_negative_nonexistent_node(self) -> None:
        flow = _simple_flow()
        llm = _make_deterministic_llm({})
        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: llm,
        )

        edge_test = EdgeTest(
            node_id="nonexistent",
            edge_id="some_edge",
            condition="test",
        )
        result = await evaluator.eval_negative_edge(edge_test, "test", "mock")
        assert result.error is not None


class TestCacheIntegration:
    """Test that caching works correctly with evaluator."""

    @pytest.mark.anyio
    async def test_cached_responses_reused(self, tmp_path: Path) -> None:
        call_count = {"n": 0}

        async def counting_llm(messages: list[dict]) -> str:
            call_count["n"] += 1
            sys_content = messages[0].get("content", "")
            if "evaluating" in sys_content:
                return json.dumps(
                    {
                        "all_required_met": True,
                        "recommended_edge_id": "wants_help",
                        "reason": "test",
                    }
                )
            return "Mock reply"

        cache = ResponseCache(tmp_path / "cache")
        flow = _simple_flow()

        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: counting_llm,
            cache=cache,
        )

        edge_test = EdgeTest(
            node_id="start",
            edge_id="wants_help",
            condition="User asks for help",
            utterances=["help me"],
        )

        # First call: hits LLM
        await evaluator.eval_edge(edge_test, "help me", "mock")
        first_count = call_count["n"]

        # Second call: should use cache
        await evaluator.eval_edge(edge_test, "help me", "mock")
        assert call_count["n"] == first_count  # no new LLM calls


class TestResponseCache:
    """Unit tests for ResponseCache."""

    def test_put_and_get(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache")
        cache.put_raw("model-a", "key1", "response1")
        assert cache.get_raw("model-a", "key1") == "response1"

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache")
        assert cache.get_raw("model-a", "missing") is None

    def test_invalidate_model(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache")
        cache.put_raw("model-a", "k1", "v1")
        cache.put_raw("model-b", "k2", "v2")
        cache.invalidate("model-a")
        assert cache.get_raw("model-a", "k1") is None
        assert cache.get_raw("model-b", "k2") == "v2"

    def test_invalidate_all(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache")
        cache.put_raw("model-a", "k1", "v1")
        cache.put_raw("model-b", "k2", "v2")
        cache.invalidate()
        assert cache.get_raw("model-a", "k1") is None
        assert cache.get_raw("model-b", "k2") is None

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache1 = ResponseCache(cache_dir)
        cache1.put_raw("model-a", "k1", "v1")

        cache2 = ResponseCache(cache_dir)
        assert cache2.get_raw("model-a", "k1") == "v1"

    def test_hash_deterministic(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        h1 = ResponseCache.hash_messages(msgs)
        h2 = ResponseCache.hash_messages(msgs)
        assert h1 == h2

    def test_hash_differs_for_different_messages(self) -> None:
        h1 = ResponseCache.hash_messages([{"role": "user", "content": "hello"}])
        h2 = ResponseCache.hash_messages([{"role": "user", "content": "world"}])
        assert h1 != h2
