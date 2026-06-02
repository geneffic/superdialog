"""Tests for EngineAdvisor -- ENGINE_BUG failure analysis."""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock

# Mock livekit modules before any project imports
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

from superdialog.machine.eval.engine_advisor import (  # noqa: E402
    EngineAdvisor,
    _format_failures_block,
    _group_key,
)
from superdialog.machine.eval.models import (  # noqa: E402
    ClassifiedFailure,
    EngineAdvice,
    FailureCategory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_engine_failure(
    node_id: str = "start",
    expected_edge: str = "go_help",
    actual_edge: str = "go_bye",
    utterance: str = "I need help",
    explanation: str = "Engine misrouted the intent",
) -> ClassifiedFailure:
    return ClassifiedFailure(
        category=FailureCategory.ENGINE_BUG,
        confidence=0.9,
        explanation=explanation,
        node_id=node_id,
        expected_edge=expected_edge,
        actual_edge=actual_edge,
        utterance=utterance,
    )


def _make_non_engine_failure(
    node_id: str = "start",
    expected_edge: str = "go_help",
) -> ClassifiedFailure:
    return ClassifiedFailure(
        category=FailureCategory.AMBIGUOUS_CONDITION,
        confidence=0.8,
        explanation="Condition is ambiguous",
        node_id=node_id,
        expected_edge=expected_edge,
        actual_edge="go_other",
        utterance="maybe",
    )


def _make_llm_fn(advice: dict) -> AsyncMock:
    """Return a mock LLM callable that returns the given advice as JSON."""
    return AsyncMock(return_value=json.dumps(advice))


SAMPLE_ADVICE = {
    "affected_file": "criteria.py",
    "description": "CriteriaJudge fails to match help-seeking intent",
    "suggested_change": "Expand the edge condition to include 'need help' patterns",
    "test_case": "Test that 'I need help' triggers go_help from start node",
}


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


class TestGroupKey:
    def test_returns_node_and_edge(self) -> None:
        failure = _make_engine_failure(node_id="intro", expected_edge="go_main")
        assert _group_key(failure) == ("intro", "go_main")

    def test_empty_fields(self) -> None:
        failure = ClassifiedFailure(
            category=FailureCategory.ENGINE_BUG,
        )
        assert _group_key(failure) == ("", "")


class TestFormatFailuresBlock:
    def test_single_failure_formatted(self) -> None:
        failure = _make_engine_failure(
            utterance="help me",
            expected_edge="go_help",
            actual_edge="go_bye",
            explanation="wrong route",
        )
        block = _format_failures_block([failure])
        assert "help me" in block
        assert "go_help" in block
        assert "go_bye" in block
        assert "wrong route" in block

    def test_multiple_failures_numbered(self) -> None:
        failures = [_make_engine_failure(utterance=f"utterance {i}") for i in range(3)]
        block = _format_failures_block(failures)
        assert "1." in block
        assert "2." in block
        assert "3." in block

    def test_empty_list_returns_empty_string(self) -> None:
        block = _format_failures_block([])
        assert block == ""


# ---------------------------------------------------------------------------
# EngineAdvisor.analyze
# ---------------------------------------------------------------------------


class TestEngineAdvisorAnalyze:
    @pytest.mark.anyio
    async def test_empty_input_returns_empty_list(self) -> None:
        llm_fn = _make_llm_fn(SAMPLE_ADVICE)
        advisor = EngineAdvisor(llm_fn=llm_fn)
        result = await advisor.analyze([])
        assert result == []
        llm_fn.assert_not_called()

    @pytest.mark.anyio
    async def test_non_engine_failures_filtered_out(self) -> None:
        failures = [
            _make_non_engine_failure(),
            ClassifiedFailure(
                category=FailureCategory.MISSING_EDGE,
                node_id="x",
                expected_edge="y",
            ),
        ]
        llm_fn = _make_llm_fn(SAMPLE_ADVICE)
        advisor = EngineAdvisor(llm_fn=llm_fn)
        result = await advisor.analyze(failures)
        assert result == []
        llm_fn.assert_not_called()

    @pytest.mark.anyio
    async def test_single_engine_failure_produces_advice(self) -> None:
        failures = [_make_engine_failure()]
        llm_fn = _make_llm_fn(SAMPLE_ADVICE)
        advisor = EngineAdvisor(llm_fn=llm_fn)
        result = await advisor.analyze(failures)
        assert len(result) == 1
        advice = result[0]
        assert isinstance(advice, EngineAdvice)
        assert advice.affected_file == "criteria.py"
        assert "CriteriaJudge" in advice.description
        assert advice.related_failures == 1

    @pytest.mark.anyio
    async def test_mixed_failures_only_engine_bugs_processed(self) -> None:
        failures = [
            _make_non_engine_failure(node_id="n1"),
            _make_engine_failure(node_id="n2", expected_edge="go_help"),
        ]
        llm_fn = _make_llm_fn(SAMPLE_ADVICE)
        advisor = EngineAdvisor(llm_fn=llm_fn)
        result = await advisor.analyze(failures)
        assert len(result) == 1
        llm_fn.assert_called_once()

    @pytest.mark.anyio
    async def test_grouping_same_node_and_edge(self) -> None:
        """Three failures on same node/edge should form one group (one LLM call)."""
        failures = [
            _make_engine_failure(
                node_id="intro",
                expected_edge="go_main",
                utterance=f"utterance {i}",
            )
            for i in range(3)
        ]
        llm_fn = _make_llm_fn(SAMPLE_ADVICE)
        advisor = EngineAdvisor(llm_fn=llm_fn)
        result = await advisor.analyze(failures)
        assert len(result) == 1
        assert result[0].related_failures == 3
        llm_fn.assert_called_once()

    @pytest.mark.anyio
    async def test_grouping_different_edges_produce_separate_advice(self) -> None:
        """Failures with different expected_edges should produce separate advice."""
        failures = [
            _make_engine_failure(
                node_id="start", expected_edge="go_help", utterance="need help"
            ),
            _make_engine_failure(
                node_id="start", expected_edge="go_bye", utterance="goodbye"
            ),
        ]
        # LLM returns different advice per call
        call_count = 0
        advices = [
            {**SAMPLE_ADVICE, "description": "advice 1"},
            {**SAMPLE_ADVICE, "description": "advice 2"},
        ]

        async def llm_fn(messages):  # type: ignore[no-untyped-def]
            nonlocal call_count
            result = json.dumps(advices[call_count])
            call_count += 1
            return result

        advisor = EngineAdvisor(llm_fn=llm_fn)
        result = await advisor.analyze(failures)
        assert len(result) == 2
        assert call_count == 2

    @pytest.mark.anyio
    async def test_grouping_different_nodes_produce_separate_advice(self) -> None:
        """Failures on different nodes should produce separate advice."""
        failures = [
            _make_engine_failure(node_id="node_a", expected_edge="go_x"),
            _make_engine_failure(node_id="node_b", expected_edge="go_x"),
        ]
        llm_fn = _make_llm_fn(SAMPLE_ADVICE)
        advisor = EngineAdvisor(llm_fn=llm_fn)
        result = await advisor.analyze(failures)
        assert len(result) == 2
        assert llm_fn.call_count == 2

    @pytest.mark.anyio
    async def test_llm_prompt_includes_utterances(self) -> None:
        """The LLM prompt should include failure utterances."""
        failures = [_make_engine_failure(utterance="please assist me")]
        llm_fn = _make_llm_fn(SAMPLE_ADVICE)
        advisor = EngineAdvisor(llm_fn=llm_fn)
        await advisor.analyze(failures)

        call_args = llm_fn.call_args
        messages = call_args[0][0]
        prompt_text = messages[0]["content"]
        assert "please assist me" in prompt_text

    @pytest.mark.anyio
    async def test_llm_prompt_includes_engine_files(self) -> None:
        """The LLM prompt should mention the engine files."""
        failures = [_make_engine_failure()]
        llm_fn = _make_llm_fn(SAMPLE_ADVICE)
        advisor = EngineAdvisor(llm_fn=llm_fn)
        await advisor.analyze(failures)

        call_args = llm_fn.call_args
        messages = call_args[0][0]
        prompt_text = messages[0]["content"]
        assert "criteria.py" in prompt_text
        assert "machine.py" in prompt_text
        assert "text_adapter.py" in prompt_text

    @pytest.mark.anyio
    async def test_unknown_affected_file_defaults_to_criteria(self) -> None:
        """If LLM returns an unknown file name, default to criteria.py."""
        bad_advice = {**SAMPLE_ADVICE, "affected_file": "unknown_file.py"}
        llm_fn = _make_llm_fn(bad_advice)
        advisor = EngineAdvisor(llm_fn=llm_fn)
        result = await advisor.analyze([_make_engine_failure()])
        assert len(result) == 1
        assert result[0].affected_file == "criteria.py"

    @pytest.mark.anyio
    async def test_json_parse_failure_returns_none_for_group(self) -> None:
        """If the LLM returns invalid JSON, that group is skipped (not raised)."""
        llm_fn = AsyncMock(return_value="not valid json {{")
        advisor = EngineAdvisor(llm_fn=llm_fn)
        result = await advisor.analyze([_make_engine_failure()])
        assert result == []

    @pytest.mark.anyio
    async def test_markdown_fenced_json_is_parsed(self) -> None:
        """JSON wrapped in markdown fences should be parsed successfully."""
        fenced = f"```json\n{json.dumps(SAMPLE_ADVICE)}\n```"
        llm_fn = AsyncMock(return_value=fenced)
        advisor = EngineAdvisor(llm_fn=llm_fn)
        result = await advisor.analyze([_make_engine_failure()])
        assert len(result) == 1
        assert result[0].affected_file == "criteria.py"

    @pytest.mark.anyio
    async def test_all_engine_file_targets_accepted(self) -> None:
        """Verify machine.py and text_adapter.py are accepted as valid files."""
        for fname in ("machine.py", "text_adapter.py"):
            advice = {**SAMPLE_ADVICE, "affected_file": fname}
            llm_fn = _make_llm_fn(advice)
            advisor = EngineAdvisor(llm_fn=llm_fn)
            result = await advisor.analyze([_make_engine_failure()])
            assert len(result) == 1
            assert result[0].affected_file == fname

    @pytest.mark.anyio
    async def test_advice_test_case_and_suggested_change_populated(self) -> None:
        llm_fn = _make_llm_fn(SAMPLE_ADVICE)
        advisor = EngineAdvisor(llm_fn=llm_fn)
        result = await advisor.analyze([_make_engine_failure()])
        assert result[0].test_case == SAMPLE_ADVICE["test_case"]
        assert result[0].suggested_change == SAMPLE_ADVICE["suggested_change"]
