"""Tests for FailureClassifier.

Classifies eval failures into root cause categories using a mock LLM.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock livekit modules before any super imports
# ---------------------------------------------------------------------------
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
from superdialog.machine.eval.failure_classifier import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    FailureClassifier,
)
from superdialog.machine.eval.models import (  # noqa: E402
    ClassifiedFailure,
    EdgeTestResult,
    EvalReport,
    FailureCategory,
    ModelScore,
    NegativeEdgeResult,
)

# ---------------------------------------------------------------------------
# Synthetic flow fixtures
# ---------------------------------------------------------------------------


def _simple_flow() -> ConversationFlow:
    """greeting -> confirm (yes) or decline (no) -> done."""
    return ConversationFlow(
        system_prompt="test flow",
        initial_node="greeting",
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                instruction="Ask the user if they are interested.",
                edges=[
                    Edge(
                        id="edge_yes",
                        condition="User says yes or agrees",
                        target_node_id="confirm",
                    ),
                    Edge(
                        id="edge_no",
                        condition="User says no or declines",
                        target_node_id="decline",
                    ),
                ],
            ),
            FlowNode(
                id="confirm",
                name="Confirm",
                instruction="Confirm the user's interest.",
                edges=[
                    Edge(
                        id="edge_done",
                        condition="User confirms",
                        target_node_id="done",
                    ),
                ],
            ),
            FlowNode(
                id="decline",
                name="Decline",
                instruction="Acknowledge the decline.",
                is_final=True,
            ),
            FlowNode(
                id="done",
                name="Done",
                instruction="Wrap up.",
                is_final=True,
            ),
        ],
    )


def _flow_with_no_edges() -> ConversationFlow:
    """Single final node with no edges."""
    return ConversationFlow(
        system_prompt="minimal",
        initial_node="only",
        nodes=[
            FlowNode(
                id="only",
                name="Only",
                instruction="just end",
                is_final=True,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------


def _make_llm_fn(response_data: dict[str, Any]):
    """Return an async LLM callable that always yields the given JSON."""

    async def _llm(messages: list[dict[str, Any]]) -> str:
        return json.dumps(response_data)

    return _llm


def _make_llm_fn_markdown(response_data: dict[str, Any]):
    """Return an async LLM callable that wraps the JSON in a markdown fence."""

    async def _llm(messages: list[dict[str, Any]]) -> str:
        body = json.dumps(response_data)
        return f"```json\n{body}\n```"

    return _llm


def _make_failing_llm():
    """Return an async LLM callable that raises RuntimeError."""

    async def _llm(messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("simulated LLM error")

    return _llm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_flow() -> ConversationFlow:
    return _simple_flow()


@pytest.fixture
def high_confidence_response() -> dict[str, Any]:
    return {
        "category": "ambiguous_condition",
        "confidence": 0.9,
        "explanation": "Both yes and no edges match vague affirmations.",
        "suggested_fix": "Add explicit keywords to the yes condition.",
    }


@pytest.fixture
def low_confidence_response() -> dict[str, Any]:
    return {
        "category": "weak_condition",
        "confidence": 0.3,
        "explanation": "Unclear failure mode.",
        "suggested_fix": "Review the condition.",
    }


def _make_edge_failure(
    node_id: str = "greeting",
    edge_id: str = "edge_yes",
    utterance: str = "maybe",
    expected_edge: str = "edge_yes",
    actual_edge: str | None = "edge_no",
) -> EdgeTestResult:
    return EdgeTestResult(
        node_id=node_id,
        edge_id=edge_id,
        utterance=utterance,
        expected_edge=expected_edge,
        actual_edge=actual_edge,
        expected_target="confirm",
        actual_target="decline",
        passed=False,
    )


def _make_negative_failure(
    node_id: str = "greeting",
    edge_id: str = "edge_yes",
    utterance: str = "I suppose",
    actual_edge: str | None = "edge_yes",
) -> NegativeEdgeResult:
    return NegativeEdgeResult(
        node_id=node_id,
        edge_id=edge_id,
        utterance=utterance,
        actual_edge=actual_edge,
        passed=False,
    )


def _make_report(
    failures: list[EdgeTestResult] | None = None,
    negative_failures: list[NegativeEdgeResult] | None = None,
) -> EvalReport:
    score = ModelScore(
        model_id="test-model",
        failures=failures or [],
        negative_failures=negative_failures or [],
    )
    return EvalReport(flow_file="test.json", models=[score])


# ---------------------------------------------------------------------------
# classify_failures — positive edge failures
# ---------------------------------------------------------------------------


class TestClassifyEdgeFailures:
    @pytest.mark.anyio
    async def test_single_failure_classified(
        self,
        simple_flow: ConversationFlow,
        high_confidence_response: dict[str, Any],
    ) -> None:
        classifier = FailureClassifier(
            simple_flow, _make_llm_fn(high_confidence_response)
        )
        report = _make_report(failures=[_make_edge_failure()])

        results = await classifier.classify_failures(report)

        assert len(results) == 1
        cf = results[0]
        assert cf.category == FailureCategory.AMBIGUOUS_CONDITION
        assert cf.confidence == pytest.approx(0.9)
        assert cf.node_id == "greeting"
        assert cf.edge_id == "edge_yes"
        assert cf.utterance == "maybe"
        assert cf.expected_edge == "edge_yes"
        assert cf.actual_edge == "edge_no"

    @pytest.mark.anyio
    async def test_multiple_failures_classified(
        self,
        simple_flow: ConversationFlow,
        high_confidence_response: dict[str, Any],
    ) -> None:
        failures = [
            _make_edge_failure(utterance="perhaps"),
            _make_edge_failure(utterance="sort of"),
        ]
        classifier = FailureClassifier(
            simple_flow, _make_llm_fn(high_confidence_response)
        )
        report = _make_report(failures=failures)

        results = await classifier.classify_failures(report)

        assert len(results) == 2

    @pytest.mark.anyio
    async def test_empty_report_returns_empty_list(
        self, simple_flow: ConversationFlow
    ) -> None:
        classifier = FailureClassifier(
            simple_flow, _make_llm_fn({"category": "unknown", "confidence": 0.8})
        )
        report = _make_report()

        results = await classifier.classify_failures(report)

        assert results == []

    @pytest.mark.anyio
    async def test_unknown_node_id_does_not_raise(
        self,
        simple_flow: ConversationFlow,
        high_confidence_response: dict[str, Any],
    ) -> None:
        """Classifier should gracefully handle a node_id missing from flow."""
        failure = _make_edge_failure(node_id="nonexistent_node")
        classifier = FailureClassifier(
            simple_flow, _make_llm_fn(high_confidence_response)
        )
        report = _make_report(failures=[failure])

        results = await classifier.classify_failures(report)

        assert len(results) == 1
        assert results[0].node_id == "nonexistent_node"

    @pytest.mark.anyio
    async def test_explanation_and_suggested_fix_propagated(
        self,
        simple_flow: ConversationFlow,
        high_confidence_response: dict[str, Any],
    ) -> None:
        classifier = FailureClassifier(
            simple_flow, _make_llm_fn(high_confidence_response)
        )
        report = _make_report(failures=[_make_edge_failure()])

        results = await classifier.classify_failures(report)

        assert "Both yes and no edges" in results[0].explanation
        assert "explicit keywords" in results[0].suggested_fix

    @pytest.mark.anyio
    async def test_markdown_fenced_response_parsed(
        self,
        simple_flow: ConversationFlow,
        high_confidence_response: dict[str, Any],
    ) -> None:
        """LLM responses wrapped in markdown fences should be handled."""
        classifier = FailureClassifier(
            simple_flow, _make_llm_fn_markdown(high_confidence_response)
        )
        report = _make_report(failures=[_make_edge_failure()])

        results = await classifier.classify_failures(report)

        assert len(results) == 1
        assert results[0].category == FailureCategory.AMBIGUOUS_CONDITION


# ---------------------------------------------------------------------------
# classify_failures — negative edge failures
# ---------------------------------------------------------------------------


class TestClassifyNegativeFailures:
    @pytest.mark.anyio
    async def test_negative_failure_classified(
        self,
        simple_flow: ConversationFlow,
        high_confidence_response: dict[str, Any],
    ) -> None:
        classifier = FailureClassifier(
            simple_flow, _make_llm_fn(high_confidence_response)
        )
        neg = _make_negative_failure()
        report = _make_report(negative_failures=[neg])

        results = await classifier.classify_failures(report)

        assert len(results) == 1
        cf = results[0]
        assert cf.node_id == "greeting"
        assert cf.edge_id == "edge_yes"
        assert cf.utterance == "I suppose"
        assert cf.expected_edge == "(should not trigger)"
        assert cf.actual_edge == "edge_yes"

    @pytest.mark.anyio
    async def test_mixed_positive_and_negative_failures(
        self,
        simple_flow: ConversationFlow,
        high_confidence_response: dict[str, Any],
    ) -> None:
        classifier = FailureClassifier(
            simple_flow, _make_llm_fn(high_confidence_response)
        )
        report = _make_report(
            failures=[_make_edge_failure()],
            negative_failures=[_make_negative_failure()],
        )

        results = await classifier.classify_failures(report)

        assert len(results) == 2

    @pytest.mark.anyio
    async def test_multiple_model_scores_all_classified(
        self,
        simple_flow: ConversationFlow,
        high_confidence_response: dict[str, Any],
    ) -> None:
        """Failures from every ModelScore in the report should be classified."""
        score_a = ModelScore(
            model_id="model-a",
            failures=[_make_edge_failure(utterance="maybe")],
        )
        score_b = ModelScore(
            model_id="model-b",
            failures=[_make_edge_failure(utterance="sort of")],
        )
        report = EvalReport(flow_file="test.json", models=[score_a, score_b])
        classifier = FailureClassifier(
            simple_flow, _make_llm_fn(high_confidence_response)
        )

        results = await classifier.classify_failures(report)

        assert len(results) == 2


# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:
    @pytest.mark.anyio
    async def test_low_confidence_sets_unknown(
        self,
        simple_flow: ConversationFlow,
        low_confidence_response: dict[str, Any],
    ) -> None:
        """Confidence below threshold must override category to UNKNOWN."""
        classifier = FailureClassifier(
            simple_flow, _make_llm_fn(low_confidence_response)
        )
        report = _make_report(failures=[_make_edge_failure()])

        results = await classifier.classify_failures(report)

        assert results[0].category == FailureCategory.UNKNOWN

    @pytest.mark.anyio
    async def test_low_confidence_preserves_explanation(
        self,
        simple_flow: ConversationFlow,
        low_confidence_response: dict[str, Any],
    ) -> None:
        """Even when overriding to UNKNOWN, explanation should be kept."""
        classifier = FailureClassifier(
            simple_flow, _make_llm_fn(low_confidence_response)
        )
        report = _make_report(failures=[_make_edge_failure()])

        results = await classifier.classify_failures(report)

        assert results[0].explanation == "Unclear failure mode."

    @pytest.mark.anyio
    async def test_confidence_exactly_at_threshold_keeps_category(
        self, simple_flow: ConversationFlow
    ) -> None:
        """Confidence exactly equal to threshold (0.5) is NOT low — kept."""
        response = {
            "category": "missing_edge",
            "confidence": CONFIDENCE_THRESHOLD,
            "explanation": "At the boundary.",
            "suggested_fix": "n/a",
        }
        classifier = FailureClassifier(simple_flow, _make_llm_fn(response))
        report = _make_report(failures=[_make_edge_failure()])

        results = await classifier.classify_failures(report)

        # The spec says confidence < 0.5 → UNKNOWN; 0.5 itself is not < 0.5
        assert results[0].category == FailureCategory.MISSING_EDGE

    @pytest.mark.anyio
    async def test_confidence_above_threshold_keeps_category(
        self, simple_flow: ConversationFlow
    ) -> None:
        response = {
            "category": "missing_edge",
            "confidence": CONFIDENCE_THRESHOLD + 0.01,
            "explanation": "Clear missing edge.",
            "suggested_fix": "Add an edge.",
        }
        classifier = FailureClassifier(simple_flow, _make_llm_fn(response))
        report = _make_report(failures=[_make_edge_failure()])

        results = await classifier.classify_failures(report)

        assert results[0].category == FailureCategory.MISSING_EDGE

    @pytest.mark.anyio
    async def test_unrecognised_category_becomes_unknown(
        self, simple_flow: ConversationFlow
    ) -> None:
        """An unrecognised category string should map to UNKNOWN."""
        response = {
            "category": "totally_made_up",
            "confidence": 0.95,
            "explanation": "No idea.",
            "suggested_fix": "?",
        }
        classifier = FailureClassifier(simple_flow, _make_llm_fn(response))
        report = _make_report(failures=[_make_edge_failure()])

        results = await classifier.classify_failures(report)

        assert results[0].category == FailureCategory.UNKNOWN

    @pytest.mark.anyio
    async def test_engine_bug_category_preserved(
        self, simple_flow: ConversationFlow
    ) -> None:
        response = {
            "category": "engine_bug",
            "confidence": 0.85,
            "explanation": "Routing logic mismatch.",
            "suggested_fix": "Fix the router.",
        }
        classifier = FailureClassifier(simple_flow, _make_llm_fn(response))
        report = _make_report(failures=[_make_edge_failure()])

        results = await classifier.classify_failures(report)

        assert results[0].category == FailureCategory.ENGINE_BUG


# ---------------------------------------------------------------------------
# split_by_fixability
# ---------------------------------------------------------------------------


class TestSplitByFixability:
    def _classified_failure(self, category: FailureCategory) -> ClassifiedFailure:
        return ClassifiedFailure(
            category=category,
            confidence=0.9,
            node_id="greeting",
            edge_id="edge_yes",
            utterance="test",
            expected_edge="edge_yes",
            actual_edge="edge_no",
        )

    def test_flow_fixable_categories(self, simple_flow: ConversationFlow) -> None:
        classifier = FailureClassifier(simple_flow, _make_llm_fn({}))
        flow_fixable_cats = [
            FailureCategory.AMBIGUOUS_CONDITION,
            FailureCategory.WEAK_CONDITION,
            FailureCategory.MISSING_EDGE,
            FailureCategory.PROMPT_CONFUSED,
            FailureCategory.SLOT_NOT_CAPTURED,
            FailureCategory.LANGUAGE_MISMATCH,
            FailureCategory.GUARDRAIL_VIOLATED,
        ]
        classified = [self._classified_failure(c) for c in flow_fixable_cats]

        flow_fixable, engine_fixable = classifier.split_by_fixability(classified)

        assert len(flow_fixable) == len(flow_fixable_cats)
        assert engine_fixable == []

    def test_engine_fixable_categories(self, simple_flow: ConversationFlow) -> None:
        classifier = FailureClassifier(simple_flow, _make_llm_fn({}))
        engine_cats = [FailureCategory.ENGINE_BUG, FailureCategory.UNKNOWN]
        classified = [self._classified_failure(c) for c in engine_cats]

        flow_fixable, engine_fixable = classifier.split_by_fixability(classified)

        assert flow_fixable == []
        assert len(engine_fixable) == 2

    def test_mixed_split(self, simple_flow: ConversationFlow) -> None:
        classifier = FailureClassifier(simple_flow, _make_llm_fn({}))
        classified = [
            self._classified_failure(FailureCategory.AMBIGUOUS_CONDITION),
            self._classified_failure(FailureCategory.ENGINE_BUG),
            self._classified_failure(FailureCategory.MISSING_EDGE),
            self._classified_failure(FailureCategory.UNKNOWN),
        ]

        flow_fixable, engine_fixable = classifier.split_by_fixability(classified)

        assert len(flow_fixable) == 2
        assert len(engine_fixable) == 2

    def test_empty_input_returns_empty_lists(
        self, simple_flow: ConversationFlow
    ) -> None:
        classifier = FailureClassifier(simple_flow, _make_llm_fn({}))

        flow_fixable, engine_fixable = classifier.split_by_fixability([])

        assert flow_fixable == []
        assert engine_fixable == []

    def test_is_flow_fixable_property_consistent(
        self, simple_flow: ConversationFlow
    ) -> None:
        """is_flow_fixable on ClassifiedFailure must mirror FailureCategory."""
        classifier = FailureClassifier(simple_flow, _make_llm_fn({}))

        for cat in FailureCategory:
            cf = ClassifiedFailure(
                category=cat,
                node_id="n",
                edge_id="e",
                utterance="u",
            )
            assert cf.is_flow_fixable == cat.is_flow_fixable

        flow_fixable, engine_fixable = classifier.split_by_fixability(
            [
                ClassifiedFailure(
                    category=FailureCategory.AMBIGUOUS_CONDITION,
                    node_id="n",
                    edge_id="e",
                    utterance="u",
                )
            ]
        )
        assert len(flow_fixable) == 1
        assert engine_fixable == []


# ---------------------------------------------------------------------------
# LLM error handling
# ---------------------------------------------------------------------------


class TestLLMErrorHandling:
    @pytest.mark.anyio
    async def test_llm_error_propagates(self, simple_flow: ConversationFlow) -> None:
        """A RuntimeError from the LLM should propagate out."""
        classifier = FailureClassifier(simple_flow, _make_failing_llm())
        report = _make_report(failures=[_make_edge_failure()])

        with pytest.raises(RuntimeError, match="simulated LLM error"):
            await classifier.classify_failures(report)

    @pytest.mark.anyio
    async def test_invalid_json_returns_unknown_category(
        self, simple_flow: ConversationFlow
    ) -> None:
        """If the LLM returns non-JSON, category should default to UNKNOWN."""

        async def _bad_llm(messages: list[dict[str, Any]]) -> str:
            return "sorry, I cannot classify this"

        classifier = FailureClassifier(simple_flow, _bad_llm)
        report = _make_report(failures=[_make_edge_failure()])

        results = await classifier.classify_failures(report)

        assert len(results) == 1
        # No category key → defaults to UNKNOWN mapping
        assert results[0].category == FailureCategory.UNKNOWN
