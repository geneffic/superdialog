"""E2E traversal tests for tech support, outbound sales, and survey flows."""

from __future__ import annotations

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

from superdialog.flow.models import ConversationFlow  # noqa: E402
from superdialog.machine.machine import DialogStateMachine  # noqa: E402
from superdialog.machine.testing.mock_adapter import MockAdapter  # noqa: E402

TEMP_DIR = Path(__file__).resolve().parents[4] / "temp"


async def _drive(flow: ConversationFlow, edges: list[str]) -> DialogStateMachine:
    """Drive a flow through the given edge sequence."""
    adapter = MockAdapter(edges)
    machine = await DialogStateMachine.from_flow(flow, adapter)
    for i in range(len(edges)):
        await machine.process_turn(f"input {i}")
    return machine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tech_support_flow() -> ConversationFlow:
    path = TEMP_DIR / "tech_support_lite_flow.json"
    if not path.exists():
        pytest.skip("tech_support_lite_flow.json not found")
    return ConversationFlow.from_json_file(path)


@pytest.fixture
def outbound_sales_flow() -> ConversationFlow:
    path = TEMP_DIR / "outbound_sales_lite_flow.json"
    if not path.exists():
        pytest.skip("outbound_sales_lite_flow.json not found")
    return ConversationFlow.from_json_file(path)


@pytest.fixture
def survey_flow() -> ConversationFlow:
    path = TEMP_DIR / "survey_feedback_lite_flow.json"
    if not path.exists():
        pytest.skip("survey_feedback_lite_flow.json not found")
    return ConversationFlow.from_json_file(path)


# ---------------------------------------------------------------------------
# TestTechSupportPaths
# ---------------------------------------------------------------------------


class TestTechSupportPaths:
    """E2E traversal tests for the tech support flow."""

    @pytest.mark.anyio
    async def test_resolved_path(self, tech_support_flow: ConversationFlow) -> None:
        """greeting -> identity -> diagnosis -> resolution -> call_close."""
        edges = [
            "greeting_to_identity",
            "identity_to_diagnosis",
            "diagnosis_to_resolution",
            "resolution_to_close",
        ]
        machine = await _drive(tech_support_flow, edges)

        assert machine.current_state == "call_close"
        assert machine.is_complete is True
        assert len(machine.context.transition_log) == 4

    @pytest.mark.anyio
    async def test_escalation_path(self, tech_support_flow: ConversationFlow) -> None:
        """greeting -> identity -> diagnosis -> escalation -> call_close."""
        edges = [
            "greeting_to_identity",
            "identity_to_diagnosis",
            "diagnosis_to_escalation",
            "escalation_to_close",
        ]
        machine = await _drive(tech_support_flow, edges)

        assert machine.current_state == "call_close"
        assert machine.is_complete is True
        assert len(machine.context.transition_log) == 4

    @pytest.mark.anyio
    async def test_billing_direct(self, tech_support_flow: ConversationFlow) -> None:
        """greeting -> billing_transfer -> call_close."""
        edges = [
            "greeting_to_billing_direct",
            "billing_to_close",
        ]
        machine = await _drive(tech_support_flow, edges)

        assert machine.current_state == "call_close"
        assert machine.is_complete is True
        assert len(machine.context.transition_log) == 2


# ---------------------------------------------------------------------------
# TestOutboundSalesPaths
# ---------------------------------------------------------------------------


class TestOutboundSalesPaths:
    """E2E traversal tests for the outbound sales flow."""

    @pytest.mark.anyio
    async def test_happy_path(self, outbound_sales_flow: ConversationFlow) -> None:
        """greeting -> discovery -> pitch -> close -> wrap_up."""
        edges = [
            "greeting_to_discovery",
            "discovery_to_pitch",
            "pitch_to_close",
            "close_to_wrap_up",
        ]
        machine = await _drive(outbound_sales_flow, edges)

        assert machine.current_state == "wrap_up"
        assert machine.is_complete is True
        assert len(machine.context.transition_log) == 4

    @pytest.mark.anyio
    async def test_objection_loop(self, outbound_sales_flow: ConversationFlow) -> None:
        """greeting -> discovery -> pitch -> objection -> close -> wrap_up."""
        edges = [
            "greeting_to_discovery",
            "discovery_to_pitch",
            "pitch_to_objection",
            "objection_to_close",
            "close_to_wrap_up",
        ]
        machine = await _drive(outbound_sales_flow, edges)

        assert machine.current_state == "wrap_up"
        assert machine.is_complete is True
        assert len(machine.context.transition_log) == 5

    @pytest.mark.anyio
    async def test_immediate_reject(
        self, outbound_sales_flow: ConversationFlow
    ) -> None:
        """greeting -> wrap_up (immediate rejection)."""
        edges = [
            "greeting_to_wrap_up",
        ]
        machine = await _drive(outbound_sales_flow, edges)

        assert machine.current_state == "wrap_up"
        assert machine.is_complete is True
        assert len(machine.context.transition_log) == 1


# ---------------------------------------------------------------------------
# TestSurveyPaths
# ---------------------------------------------------------------------------


class TestSurveyPaths:
    """E2E traversal tests for the survey feedback flow."""

    @pytest.mark.anyio
    async def test_complete_survey(self, survey_flow: ConversationFlow) -> None:
        """Full survey: greeting -> A -> B -> C -> thank_you -> close."""
        edges = [
            "greeting_to_section_a",
            "section_a_to_section_b",
            "section_b_to_section_c",
            "section_c_to_thank_you",
            "thank_you_to_close",
        ]
        machine = await _drive(survey_flow, edges)

        assert machine.current_state == "close_call"
        assert machine.is_complete is True
        assert len(machine.context.transition_log) == 5

    @pytest.mark.anyio
    async def test_early_exit(self, survey_flow: ConversationFlow) -> None:
        """Early exit: greeting -> section_a -> close_call."""
        edges = [
            "greeting_to_section_a",
            "section_a_to_close",
        ]
        machine = await _drive(survey_flow, edges)

        assert machine.current_state == "close_call"
        assert machine.is_complete is True
        assert len(machine.context.transition_log) == 2

    @pytest.mark.anyio
    async def test_callback_then_proceed(self, survey_flow: ConversationFlow) -> None:
        """Callback then full survey: greeting -> callback -> A -> B -> C -> thank_you -> close."""
        edges = [
            "greeting_to_callback",
            "callback_to_section_a",
            "section_a_to_section_b",
            "section_b_to_section_c",
            "section_c_to_thank_you",
            "thank_you_to_close",
        ]
        machine = await _drive(survey_flow, edges)

        assert machine.current_state == "close_call"
        assert machine.is_complete is True
        assert len(machine.context.transition_log) == 6
