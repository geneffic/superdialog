"""Tier 3: Multi-model scorecard tests.

Tests that run eval corpus against multiple LLMs and produce
a comparative report. Skipped in CI unless --run-live-eval flag
is set.

Usage:
    pytest test_multi_model.py -v --run-live-eval
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
from superdialog.machine.eval.evaluator import FlowEvaluator  # noqa: E402
from superdialog.machine.eval.models import (  # noqa: E402
    EdgeTest,
    EvalReport,
    TestCorpus,
)


def _simple_flow() -> ConversationFlow:
    return ConversationFlow(
        system_prompt="You are a helpful assistant.",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="Greet and route.",
                edges=[
                    Edge(
                        id="go_a",
                        condition="User says A",
                        target_node_id="a",
                    ),
                    Edge(
                        id="go_b",
                        condition="User says B",
                        target_node_id="b",
                    ),
                ],
            ),
            FlowNode(id="a", name="A", static_text="A!", is_final=True),
            FlowNode(id="b", name="B", static_text="B!", is_final=True),
        ],
    )


def _make_model_llm(model_id: str, behavior: dict[str, str]):
    """Create mock LLM that varies behavior by model_id."""

    async def llm(messages: list[dict]) -> str:
        sys_content = messages[0].get("content", "")
        if "evaluating" in sys_content:
            edge_id = behavior.get(model_id, "go_a")
            return json.dumps(
                {
                    "all_required_met": True,
                    "recommended_edge_id": edge_id,
                    "reason": f"model {model_id} choice",
                }
            )
        return f"Reply from {model_id}"

    return llm


class TestMultiModelScorecard:
    """Test multi-model comparison with mock LLMs."""

    @pytest.mark.anyio
    async def test_two_models_compared(self) -> None:
        flow = _simple_flow()
        behaviors = {"model-good": "go_a", "model-bad": "go_b"}

        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda mid: _make_model_llm(mid, behaviors),
        )

        corpus = TestCorpus(
            flow_file="test.json",
            edge_tests=[
                EdgeTest(
                    node_id="start",
                    edge_id="go_a",
                    condition="User says A",
                    utterances=["A please"],
                ),
            ],
        )

        report = await evaluator.eval_corpus(corpus, ["model-good", "model-bad"])

        assert len(report.models) == 2

        good = next(m for m in report.models if m.model_id == "model-good")
        bad = next(m for m in report.models if m.model_id == "model-bad")

        assert good.edge_accuracy == 1.0
        assert bad.edge_accuracy == 0.0
        assert good.edges_passed == 1
        assert bad.edges_passed == 0

    @pytest.mark.anyio
    async def test_report_regression_detection(self) -> None:
        # Baseline: model was at 100%
        baseline = EvalReport(
            flow_file="test.json",
            models=[
                _mock_score(
                    "model-x",
                    edge_accuracy=1.0,
                    path_completion=1.0,
                )
            ],
        )

        # Current: model dropped to 50%
        current = EvalReport(
            flow_file="test.json",
            models=[
                _mock_score(
                    "model-x",
                    edge_accuracy=0.5,
                    path_completion=0.8,
                )
            ],
        )

        regressions = current.compare_baseline(baseline)
        assert len(regressions) == 2
        assert any("edge accuracy dropped" in r for r in regressions)
        assert any("path completion dropped" in r for r in regressions)

    @pytest.mark.anyio
    async def test_report_no_regression(self) -> None:
        baseline = EvalReport(
            flow_file="test.json",
            models=[
                _mock_score(
                    "model-x",
                    edge_accuracy=0.8,
                    path_completion=0.9,
                )
            ],
        )
        current = EvalReport(
            flow_file="test.json",
            models=[
                _mock_score(
                    "model-x",
                    edge_accuracy=0.9,
                    path_completion=1.0,
                )
            ],
        )
        regressions = current.compare_baseline(baseline)
        assert regressions == []

    @pytest.mark.anyio
    async def test_negative_accuracy_regression(self) -> None:
        """Detect regressions in negative accuracy."""
        baseline = EvalReport(
            flow_file="test.json",
            models=[
                _mock_score(
                    "model-x",
                    edge_accuracy=1.0,
                    path_completion=1.0,
                    negative_accuracy=1.0,
                )
            ],
        )
        current = EvalReport(
            flow_file="test.json",
            models=[
                _mock_score(
                    "model-x",
                    edge_accuracy=1.0,
                    path_completion=1.0,
                    negative_accuracy=0.5,
                )
            ],
        )
        regressions = current.compare_baseline(baseline)
        assert len(regressions) == 1
        assert "negative accuracy dropped" in regressions[0]

    @pytest.mark.anyio
    async def test_edge_coverage_reported(self) -> None:
        flow = _simple_flow()

        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda mid: _make_model_llm(mid, {"m": "go_a"}),
        )

        # Only test one of two edges
        corpus = TestCorpus(
            flow_file="test.json",
            edge_tests=[
                EdgeTest(
                    node_id="start",
                    edge_id="go_a",
                    condition="User says A",
                    utterances=["A"],
                ),
            ],
        )

        report = await evaluator.eval_corpus(corpus, ["m"])
        assert report.edge_coverage == 0.5  # 1 of 2 edges
        assert "go_b" in report.uncovered_edges


def _mock_score(
    model_id: str,
    edge_accuracy: float = 1.0,
    path_completion: float = 1.0,
    negative_accuracy: float = 1.0,
):
    """Helper to create a ModelScore for regression tests."""
    from superdialog.machine.eval.models import ModelScore

    return ModelScore(
        model_id=model_id,
        edge_accuracy=edge_accuracy,
        path_completion=path_completion,
        negative_accuracy=negative_accuracy,
        edges_passed=int(edge_accuracy * 10),
        edges_total=10,
        negatives_passed=int(negative_accuracy * 10),
        negatives_total=10,
        paths_passed=int(path_completion * 5),
        paths_total=5,
    )
