"""Tests for FlowOptimizer -- DSPy-powered flow improvement.

Tests the optimizer's issue extraction and scoring logic.
DSPy integration tests are skipped unless --run-live-eval flag is set.
"""

from __future__ import annotations

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

from superdialog.machine.eval.flow_optimizer import (  # noqa: E402
    FlowIssue,
    FlowOptimizer,
    OptimizationResult,
)
from superdialog.machine.eval.models import (  # noqa: E402
    ClassifiedFailure,
    EdgeTestResult,
    EvalReport,
    FailureCategory,
    ModelScore,
    NegativeEdgeResult,
)


class TestIssueExtraction:
    def test_extracts_positive_failures(self) -> None:
        report = EvalReport(
            flow_file="test.json",
            models=[
                ModelScore(
                    model_id="test",
                    edge_accuracy=0.5,
                    edges_passed=1,
                    edges_total=2,
                    failures=[
                        EdgeTestResult(
                            node_id="start",
                            edge_id="wants_help",
                            utterance="I need help",
                            expected_edge="wants_help",
                            actual_edge="wants_bye",
                            expected_target="help",
                            actual_target="end",
                        ),
                    ],
                ),
            ],
        )

        optimizer = FlowOptimizer.__new__(FlowOptimizer)
        issues = optimizer.extract_issues(report)
        assert len(issues) == 1
        assert issues[0].node_id == "start"
        assert issues[0].issue_type == "wrong_transition"
        assert issues[0].expected == "wants_help"
        assert issues[0].actual == "wants_bye"

    def test_extracts_negative_failures(self) -> None:
        report = EvalReport(
            flow_file="test.json",
            models=[
                ModelScore(
                    model_id="test",
                    edge_accuracy=1.0,
                    negative_accuracy=0.0,
                    negatives_passed=0,
                    negatives_total=1,
                    negative_failures=[
                        NegativeEdgeResult(
                            node_id="start",
                            edge_id="wants_help",
                            utterance="bye bye",
                            actual_edge="wants_help",
                            passed=False,
                        ),
                    ],
                ),
            ],
        )

        optimizer = FlowOptimizer.__new__(FlowOptimizer)
        issues = optimizer.extract_issues(report)
        assert len(issues) == 1
        assert issues[0].issue_type == "false_positive"

    def test_no_issues_from_clean_report(self) -> None:
        report = EvalReport(
            flow_file="test.json",
            models=[
                ModelScore(
                    model_id="test",
                    edge_accuracy=1.0,
                    negative_accuracy=1.0,
                    edges_passed=10,
                    edges_total=10,
                ),
            ],
        )

        optimizer = FlowOptimizer.__new__(FlowOptimizer)
        issues = optimizer.extract_issues(report)
        assert issues == []


class TestScoreExtraction:
    def test_score_from_edges_only(self) -> None:
        report = EvalReport(
            flow_file="test.json",
            models=[
                ModelScore(
                    model_id="test",
                    edge_accuracy=0.8,
                    edges_passed=8,
                    edges_total=10,
                ),
            ],
        )
        score = FlowOptimizer._score_from_report(report)
        assert score == pytest.approx(0.8)

    def test_score_with_paths(self) -> None:
        report = EvalReport(
            flow_file="test.json",
            models=[
                ModelScore(
                    model_id="test",
                    edge_accuracy=0.8,
                    path_completion=0.6,
                    edges_passed=8,
                    edges_total=10,
                    paths_passed=3,
                    paths_total=5,
                ),
            ],
        )
        score = FlowOptimizer._score_from_report(report)
        # Average of 0.8 and 0.6
        assert score == pytest.approx(0.7)

    def test_score_empty_report(self) -> None:
        report = EvalReport(flow_file="test.json", models=[])
        score = FlowOptimizer._score_from_report(report)
        assert score == 0.0


class TestOptimizationResult:
    def test_result_model(self) -> None:
        result = OptimizationResult(
            original_score=0.5,
            improved_score=0.9,
            iterations=2,
            changes_made=["Fixed node instruction"],
            improved_flow={"nodes": []},
        )
        assert result.original_score == 0.5
        assert result.improved_score == 0.9
        assert result.iterations == 2
        assert len(result.changes_made) == 1


class TestFlowIssue:
    def test_issue_model(self) -> None:
        issue = FlowIssue(
            node_id="start",
            edge_id="go_a",
            issue_type="wrong_transition",
            description="Expected go_a but got go_b",
            utterance="test input",
            expected="go_a",
            actual="go_b",
        )
        assert issue.node_id == "start"
        assert issue.issue_type == "wrong_transition"


class TestClassifiedFailureExtraction:
    """Tests for extract_issues_from_classified (task 6.1–6.3)."""

    def _make_classified(
        self,
        category: FailureCategory,
        node_id: str = "node_a",
        edge_id: str = "edge_1",
    ) -> ClassifiedFailure:
        return ClassifiedFailure(
            category=category,
            confidence=0.9,
            explanation=f"Test {category.value}",
            node_id=node_id,
            edge_id=edge_id,
            utterance="test utterance",
            expected_edge="edge_1",
            actual_edge="edge_2",
            suggested_fix="fix it",
        )

    def test_skips_engine_bug(self) -> None:
        failures = [
            self._make_classified(FailureCategory.ENGINE_BUG),
            self._make_classified(FailureCategory.AMBIGUOUS_CONDITION),
        ]
        optimizer = FlowOptimizer.__new__(FlowOptimizer)
        issues = optimizer.extract_issues_from_classified(failures)
        assert len(issues) == 1
        assert "ambiguous_condition" in issues[0].issue_type

    def test_skips_unknown(self) -> None:
        failures = [self._make_classified(FailureCategory.UNKNOWN)]
        optimizer = FlowOptimizer.__new__(FlowOptimizer)
        issues = optimizer.extract_issues_from_classified(failures)
        assert len(issues) == 0

    def test_maps_category_to_strategy(self) -> None:
        failures = [
            self._make_classified(FailureCategory.AMBIGUOUS_CONDITION),
            self._make_classified(FailureCategory.WEAK_CONDITION),
            self._make_classified(FailureCategory.PROMPT_CONFUSED),
        ]
        optimizer = FlowOptimizer.__new__(FlowOptimizer)
        issues = optimizer.extract_issues_from_classified(failures)
        assert len(issues) == 3
        assert "disambiguate" in issues[0].issue_type
        assert "strengthen" in issues[1].issue_type
        assert "rewrite_instruction" in issues[2].issue_type

    def test_includes_suggested_fix_in_description(self) -> None:
        failures = [self._make_classified(FailureCategory.WEAK_CONDITION)]
        optimizer = FlowOptimizer.__new__(FlowOptimizer)
        issues = optimizer.extract_issues_from_classified(failures)
        assert "fix it" in issues[0].description

    def test_all_flow_fixable_categories(self) -> None:
        """All flow-fixable categories produce issues."""
        flow_fixable = [
            FailureCategory.AMBIGUOUS_CONDITION,
            FailureCategory.WEAK_CONDITION,
            FailureCategory.MISSING_EDGE,
            FailureCategory.PROMPT_CONFUSED,
            FailureCategory.SLOT_NOT_CAPTURED,
            FailureCategory.LANGUAGE_MISMATCH,
            FailureCategory.GUARDRAIL_VIOLATED,
        ]
        failures = [self._make_classified(c) for c in flow_fixable]
        optimizer = FlowOptimizer.__new__(FlowOptimizer)
        issues = optimizer.extract_issues_from_classified(failures)
        assert len(issues) == len(flow_fixable)

    def test_empty_classified_list(self) -> None:
        optimizer = FlowOptimizer.__new__(FlowOptimizer)
        issues = optimizer.extract_issues_from_classified([])
        assert issues == []


class TestRegressionDetection:
    """Tests for regression revert logic (task 6.4)."""

    @pytest.mark.anyio
    async def test_optimize_reverts_on_regression(self) -> None:
        """If re-eval shows lower score, revert to best flow."""
        from unittest.mock import patch

        optimizer = FlowOptimizer(max_iterations=3)

        # Mock the initial report
        report = EvalReport(
            flow_file="test.json",
            models=[
                ModelScore(
                    model_id="test",
                    edge_accuracy=0.6,
                    edges_passed=6,
                    edges_total=10,
                    failures=[
                        EdgeTestResult(
                            node_id="s",
                            edge_id="e",
                            utterance="x",
                            expected_edge="e",
                            actual_edge="f",
                            expected_target="t",
                        )
                    ],
                )
            ],
        )

        # Simulate: iter1 improves to 0.8, iter2 regresses to 0.5
        call_count = 0

        async def mock_eval_fn(flow: object) -> EvalReport:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return EvalReport(
                    flow_file="test.json",
                    models=[
                        ModelScore(
                            model_id="test",
                            edge_accuracy=0.8,
                            edges_passed=8,
                            edges_total=10,
                            failures=[
                                EdgeTestResult(
                                    node_id="s",
                                    edge_id="e",
                                    utterance="x",
                                    expected_edge="e",
                                    actual_edge="f",
                                    expected_target="t",
                                )
                            ],
                        )
                    ],
                )
            # Second iteration regresses
            return EvalReport(
                flow_file="test.json",
                models=[
                    ModelScore(
                        model_id="test",
                        edge_accuracy=0.5,
                        edges_passed=5,
                        edges_total=10,
                        failures=[
                            EdgeTestResult(
                                node_id="s",
                                edge_id="e",
                                utterance="x",
                                expected_edge="e",
                                actual_edge="f",
                                expected_target="t",
                            )
                        ],
                    )
                ],
            )

        # Mock optimize_step to return flow with a marker
        iter_count = 0

        async def mock_step(
            flow: object, issues: list[FlowIssue]
        ) -> tuple[object, list[str]]:
            nonlocal iter_count
            iter_count += 1
            mock_flow = MagicMock()
            mock_flow.model_dump.return_value = {"iter": iter_count}
            return mock_flow, [f"change_{iter_count}"]

        with patch.object(optimizer, "optimize_step", side_effect=mock_step):
            result = await optimizer.optimize(
                flow=MagicMock(),
                report=report,
                eval_fn=mock_eval_fn,
            )

        # Best score should be 0.8 (from iter1, not regressed 0.5)
        assert result.improved_score == pytest.approx(0.8)
        # The improved flow should be from iter1 (the best)
        assert result.improved_flow == {"iter": 1}
