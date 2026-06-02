"""Tests for RLLoop -- eval→classify→patch→re-eval RL loop.

All external dependencies (livekit, FlowEvaluator, FailureClassifier,
FlowOptimizer, EngineAdvisor, CorpusGenerator) are mocked so the
tests exercise only the loop orchestration logic.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
from superdialog.machine.eval.models import (  # noqa: E402
    ClassifiedFailure,
    EdgeTestResult,
    EngineAdvice,
    EvalReport,
    FailureCategory,
    ModelScore,
    RLResult,
    TestCorpus,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _simple_flow() -> ConversationFlow:
    """Minimal 3-node flow used across tests."""
    return ConversationFlow(
        system_prompt="Test assistant.",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="Greet the user.",
                edges=[
                    Edge(
                        id="go_help",
                        condition="User needs help",
                        target_node_id="help",
                    ),
                    Edge(
                        id="go_end",
                        condition="User says goodbye",
                        target_node_id="end",
                    ),
                ],
            ),
            FlowNode(
                id="help",
                name="Help",
                instruction="Assist the user.",
                edges=[
                    Edge(
                        id="done",
                        condition="Issue resolved",
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


def _minimal_corpus(flow: ConversationFlow) -> TestCorpus:
    """Pre-built corpus so CorpusGenerator is never called."""
    return TestCorpus(
        flow_file="test.json",
        generated_by="fixture",
        reviewed=True,
        edge_tests=[],
        path_tests=[],
    )


def _report(score: float, flow_file: str = "test.json") -> EvalReport:
    """Create an EvalReport with a single model at the given edge_accuracy."""
    edges_passed = int(score * 10)
    return EvalReport(
        flow_file=flow_file,
        models=[
            ModelScore(
                model_id="mock-model",
                edge_accuracy=score,
                edges_passed=edges_passed,
                edges_total=10,
                failures=(
                    [
                        EdgeTestResult(
                            node_id="start",
                            edge_id="go_help",
                            utterance="I need help",
                            expected_edge="go_help",
                            actual_edge="go_end",
                            expected_target="help",
                            passed=False,
                            model_id="mock-model",
                        )
                    ]
                    if score < 1.0
                    else []
                ),
            )
        ],
    )


def _flow_fixable_failure() -> ClassifiedFailure:
    return ClassifiedFailure(
        category=FailureCategory.AMBIGUOUS_CONDITION,
        confidence=0.9,
        explanation="Condition is too broad",
        node_id="start",
        edge_id="go_help",
        utterance="I need help",
        expected_edge="go_help",
        actual_edge="go_end",
        suggested_fix="Make condition more specific",
    )


def _engine_failure() -> ClassifiedFailure:
    return ClassifiedFailure(
        category=FailureCategory.ENGINE_BUG,
        confidence=0.8,
        explanation="Edge scoring logic incorrect",
        node_id="start",
        edge_id="go_help",
        utterance="I need help",
        expected_edge="go_help",
        actual_edge="go_end",
    )


def _make_llm_factory() -> Any:
    async def _llm(messages: list[dict]) -> str:
        return "mock response"

    def factory(model_id: str) -> Any:
        return _llm

    return factory


# ---------------------------------------------------------------------------
# Utility: build RLLoop with all heavy deps mocked
# ---------------------------------------------------------------------------


def _patch_targets() -> list[str]:
    """Return import paths to patch for isolation."""
    base = "super.core.voice.dialog_machine.eval.rl_loop"
    return [
        f"{base}.FlowEvaluator",
        f"{base}.CorpusGenerator",
        f"{base}.FlowOptimizer",
        f"{base}.FailureClassifier",
        f"{base}.EngineAdvisor",
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRLLoopTargetReached:
    """Stop immediately when initial eval already meets target accuracy."""

    @pytest.mark.anyio
    async def test_stops_when_target_already_met(self) -> None:
        from superdialog.machine.eval.rl_loop import RLLoop

        flow = _simple_flow()
        corpus = _minimal_corpus(flow)
        perfect_report = _report(1.0)

        mock_evaluator = MagicMock()
        mock_evaluator.eval_corpus = AsyncMock(return_value=perfect_report)

        with patch(
            "super.core.voice.dialog_machine.eval.rl_loop.FlowEvaluator",
            return_value=mock_evaluator,
        ):
            loop = RLLoop(
                flow=flow,
                llm_factory=_make_llm_factory(),
                model_ids=["mock-model"],
                max_iterations=5,
                target_accuracy=0.95,
            )
            result = await loop.run(corpus=corpus)

        assert isinstance(result, RLResult)
        # Only the initial report — no iterations ran
        assert result.iterations == 0
        assert result.original_score == pytest.approx(1.0)
        assert result.improved_score == pytest.approx(1.0)
        # eval_corpus called exactly once (initial)
        mock_evaluator.eval_corpus.assert_called_once()


class TestRLLoopTwoIterations:
    """Simulate two iterations of genuine improvement."""

    @pytest.mark.anyio
    async def test_two_iterations_of_improvement(self) -> None:
        from superdialog.machine.eval.rl_loop import RLLoop

        flow = _simple_flow()
        corpus = _minimal_corpus(flow)

        # Scores: 0.7 → 0.85 → 0.96 (target reached)
        reports = [_report(0.7), _report(0.85), _report(0.96)]
        report_iter = iter(reports)

        mock_evaluator_instances: list[MagicMock] = []

        def make_evaluator(**kwargs: Any) -> MagicMock:
            m = MagicMock()
            m.eval_corpus = AsyncMock(return_value=next(report_iter))
            mock_evaluator_instances.append(m)
            return m

        mock_classifier = MagicMock()
        mock_classifier.classify_report = AsyncMock(
            return_value=[_flow_fixable_failure()]
        )

        mock_optimizer = MagicMock()
        mock_optimizer.optimize_step = AsyncMock(
            return_value=(flow, ["Fixed ambiguous condition"])
        )

        with (
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.FlowEvaluator",
                side_effect=make_evaluator,
            ),
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.FailureClassifier",
                return_value=mock_classifier,
            ),
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.FlowOptimizer",
                return_value=mock_optimizer,
            ),
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.EngineAdvisor",
                None,
            ),
        ):
            loop = RLLoop(
                flow=flow,
                llm_factory=_make_llm_factory(),
                model_ids=["mock-model"],
                max_iterations=5,
                target_accuracy=0.95,
            )
            result = await loop.run(corpus=corpus)

        assert result.iterations == 2
        assert result.original_score == pytest.approx(0.7)
        assert result.improved_score == pytest.approx(0.96)
        assert len(result.iteration_reports) == 3  # initial + 2
        assert "Fixed ambiguous condition" in result.changes_made


class TestRLLoopNoImprovement:
    """Stop when the optimizer makes a change but score stays the same."""

    @pytest.mark.anyio
    async def test_stops_on_score_drop(self) -> None:
        from superdialog.machine.eval.rl_loop import RLLoop

        flow = _simple_flow()
        corpus = _minimal_corpus(flow)

        # Initial 0.7, then after patch score drops to 0.5
        reports = [_report(0.7), _report(0.5)]
        report_iter = iter(reports)

        def make_evaluator(**kwargs: Any) -> MagicMock:
            m = MagicMock()
            m.eval_corpus = AsyncMock(return_value=next(report_iter))
            return m

        mock_classifier = MagicMock()
        mock_classifier.classify_report = AsyncMock(
            return_value=[_flow_fixable_failure()]
        )

        mock_optimizer = MagicMock()
        mock_optimizer.optimize_step = AsyncMock(return_value=(flow, ["Applied fix"]))

        with (
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.FlowEvaluator",
                side_effect=make_evaluator,
            ),
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.FailureClassifier",
                return_value=mock_classifier,
            ),
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.FlowOptimizer",
                return_value=mock_optimizer,
            ),
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.EngineAdvisor",
                None,
            ),
        ):
            loop = RLLoop(
                flow=flow,
                llm_factory=_make_llm_factory(),
                model_ids=["mock-model"],
                max_iterations=5,
                target_accuracy=0.95,
            )
            result = await loop.run(corpus=corpus)

        # Loop stopped after 1 iteration (score dropped → reverted)
        assert result.iterations == 1
        # Best score never improved
        assert result.improved_score == pytest.approx(0.7)
        # Revert notice in changes
        assert any("Reverted" in c for c in result.changes_made)


class TestRLLoopOnlyEngineBugs:
    """Stop when all remaining failures are ENGINE_BUG (not flow-fixable)."""

    @pytest.mark.anyio
    async def test_stops_when_only_engine_bugs(self) -> None:
        from superdialog.machine.eval.rl_loop import RLLoop

        flow = _simple_flow()
        corpus = _minimal_corpus(flow)

        initial_report = _report(0.7)

        mock_evaluator = MagicMock()
        mock_evaluator.eval_corpus = AsyncMock(return_value=initial_report)

        mock_classifier = MagicMock()
        # All failures classified as ENGINE_BUG (not flow-fixable)
        mock_classifier.classify_report = AsyncMock(return_value=[_engine_failure()])

        mock_engine_advice = [
            EngineAdvice(
                affected_file="runner.py",
                description="Fix edge selection",
                suggested_change="Use max instead of argmax",
                related_failures=1,
            )
        ]
        mock_advisor = MagicMock()
        mock_advisor.advise = AsyncMock(return_value=mock_engine_advice)

        with (
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.FlowEvaluator",
                return_value=mock_evaluator,
            ),
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.FailureClassifier",
                return_value=mock_classifier,
            ),
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.FlowOptimizer",
            ) as _mock_opt_cls,
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.EngineAdvisor",
                return_value=mock_advisor,
            ),
        ):
            loop = RLLoop(
                flow=flow,
                llm_factory=_make_llm_factory(),
                model_ids=["mock-model"],
                max_iterations=5,
                target_accuracy=0.95,
            )
            result = await loop.run(corpus=corpus)

        # FlowOptimizer.optimize_step should never be called
        _mock_opt_cls.return_value.optimize_step.assert_not_called()

        # Engine advice should be populated
        assert len(result.engine_advice) == 1
        assert result.engine_advice[0].affected_file == "runner.py"

        # Only 0 iterations: loop exited on "no flow-fixable" check
        assert result.iterations == 0


class TestRLLoopScoreHelper:
    """Unit tests for the _score helper."""

    def test_score_single_model(self) -> None:
        from superdialog.machine.eval.rl_loop import RLLoop

        flow = _simple_flow()
        loop = RLLoop.__new__(RLLoop)
        report = _report(0.8)
        assert loop._score(report) == pytest.approx(0.8)

    def test_score_multi_model(self) -> None:
        from superdialog.machine.eval.rl_loop import RLLoop

        flow = _simple_flow()
        loop = RLLoop.__new__(RLLoop)
        report = EvalReport(
            flow_file="test.json",
            models=[
                ModelScore(model_id="m1", edge_accuracy=0.8, edges_total=10),
                ModelScore(model_id="m2", edge_accuracy=0.6, edges_total=10),
            ],
        )
        assert loop._score(report) == pytest.approx(0.7)

    def test_score_empty_report(self) -> None:
        from superdialog.machine.eval.rl_loop import RLLoop

        loop = RLLoop.__new__(RLLoop)
        report = EvalReport(flow_file="test.json", models=[])
        assert loop._score(report) == 0.0


class TestRLLoopCorpusGeneration:
    """When no corpus is provided, CorpusGenerator should be used."""

    @pytest.mark.anyio
    async def test_generates_corpus_when_none_provided(self) -> None:
        from superdialog.machine.eval.rl_loop import RLLoop

        flow = _simple_flow()
        generated_corpus = _minimal_corpus(flow)
        perfect_report = _report(1.0)

        mock_generator = MagicMock()
        mock_generator.generate_corpus = AsyncMock(return_value=generated_corpus)

        mock_evaluator = MagicMock()
        mock_evaluator.eval_corpus = AsyncMock(return_value=perfect_report)

        with (
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.CorpusGenerator",
                return_value=mock_generator,
            ),
            patch(
                "super.core.voice.dialog_machine.eval.rl_loop.FlowEvaluator",
                return_value=mock_evaluator,
            ),
        ):
            loop = RLLoop(
                flow=flow,
                llm_factory=_make_llm_factory(),
                model_ids=["mock-model"],
                max_iterations=3,
                target_accuracy=0.95,
            )
            result = await loop.run(corpus=None)

        mock_generator.generate_corpus.assert_called_once()
        assert result.original_score == pytest.approx(1.0)
