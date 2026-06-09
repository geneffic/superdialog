"""Tests for SessionAuditor — all 4 layers."""
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
from superdialog.machine.eval.models import AuditReport  # noqa: E402
from superdialog.machine.eval.session_auditor import SessionAuditor  # noqa: E402


def _simple_flow() -> ConversationFlow:
    return ConversationFlow(
        system_prompt="You are a helpful assistant.",
        initial_node="greeting",
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                instruction="Greet the user.",
                edges=[
                    Edge(id="e_bye", condition="User says goodbye", target_node_id="farewell"),
                    Edge(id="e_help", condition="User asks for help", target_node_id="farewell"),
                ],
            ),
            FlowNode(
                id="farewell",
                name="Farewell",
                static_text="Goodbye!",
                is_final=True,
            ),
        ],
    )


def _valid_traversal() -> dict:
    return {
        "session_id": "test_001",
        "flow_file": "test.json",
        "is_complete": True,
        "traversal": [
            {
                "step": 1,
                "from_node": None,
                "to_node": "greeting",
                "edge_id": None,
                "bot_message": "Hello! How can I help?",
                "user_message": None,
                "node_instruction": "Greet the user.",
                "criteria": None,
            },
            {
                "step": 2,
                "from_node": "greeting",
                "to_node": "farewell",
                "edge_id": "e_bye",
                "bot_message": "Goodbye! Have a great day.",
                "user_message": "Bye!",
                "node_instruction": "Greet the user.",
                "criteria": {"met": True, "edge_id": "e_bye", "criteria_map": {"name_given": True}},
            },
        ],
    }


class TestLayer1PathValidity:
    @pytest.mark.anyio
    async def test_valid_path_passes(self) -> None:
        auditor = SessionAuditor(flow=_simple_flow())
        report = await auditor.audit(_valid_traversal())
        assert report.path_valid is True
        assert report.path_violations == []

    @pytest.mark.anyio
    async def test_unknown_edge_flagged(self) -> None:
        traversal = _valid_traversal()
        traversal["traversal"][1]["edge_id"] = "nonexistent_edge"
        auditor = SessionAuditor(flow=_simple_flow())
        report = await auditor.audit(traversal)
        assert report.path_valid is False
        assert len(report.path_violations) == 1
        assert "not found" in report.path_violations[0].reason

    @pytest.mark.anyio
    async def test_wrong_target_flagged(self) -> None:
        traversal = _valid_traversal()
        traversal["traversal"][1]["to_node"] = "other"
        auditor = SessionAuditor(flow=_simple_flow())
        report = await auditor.audit(traversal)
        assert report.path_valid is False
        assert any("targets" in v.reason for v in report.path_violations)

    @pytest.mark.anyio
    async def test_first_step_no_edge_always_valid(self) -> None:
        traversal = _valid_traversal()
        traversal["traversal"] = [traversal["traversal"][0]]
        auditor = SessionAuditor(flow=_simple_flow())
        report = await auditor.audit(traversal)
        assert report.path_valid is True


class TestLayer2EdgeAccuracy:
    @pytest.mark.anyio
    async def test_correct_edge_verdict(self) -> None:
        async def mock_llm(messages):
            return json.dumps({"correct": True, "confidence": "high", "preferred_edge": None, "reason": "correct"})

        auditor = SessionAuditor(flow=_simple_flow(), llm_fn=mock_llm)
        report = await auditor.audit(_valid_traversal())
        assert len(report.edge_verdicts) == 1
        assert report.edge_verdicts[0].correct is True
        assert report.edge_accuracy == 1.0

    @pytest.mark.anyio
    async def test_wrong_edge_verdict(self) -> None:
        async def mock_llm(messages):
            return json.dumps({"correct": False, "confidence": "high", "preferred_edge": "e_help", "reason": "wrong"})

        auditor = SessionAuditor(flow=_simple_flow(), llm_fn=mock_llm)
        report = await auditor.audit(_valid_traversal())
        assert report.edge_verdicts[0].correct is False
        assert report.edge_verdicts[0].preferred_edge == "e_help"
        assert report.edge_accuracy == 0.0

    @pytest.mark.anyio
    async def test_no_llm_skips_layer2(self) -> None:
        auditor = SessionAuditor(flow=_simple_flow(), llm_fn=None)
        report = await auditor.audit(_valid_traversal())
        assert report.edge_verdicts == []
        assert report.edge_accuracy == 0.0


class TestLayer3ResponseQuality:
    @pytest.mark.anyio
    async def test_routing_leak_detected_by_llm(self) -> None:
        async def mock_llm(messages):
            return json.dumps({"score": 1, "routing_leak": True, "issues": ["leaked routing logic"]})

        traversal = _valid_traversal()
        traversal["traversal"][0]["bot_message"] = "If user says goodbye, transition to farewell node"
        auditor = SessionAuditor(flow=_simple_flow(), llm_fn=mock_llm)
        report = await auditor.audit(traversal)
        assert len(report.routing_leaks) >= 1

    @pytest.mark.anyio
    async def test_clean_response_no_leak(self) -> None:
        async def mock_llm(messages):
            return json.dumps({"score": 5, "routing_leak": False, "issues": []})

        auditor = SessionAuditor(flow=_simple_flow(), llm_fn=mock_llm)
        report = await auditor.audit(_valid_traversal())
        assert report.routing_leaks == []

    @pytest.mark.anyio
    async def test_no_llm_skips_layer3(self) -> None:
        auditor = SessionAuditor(flow=_simple_flow(), llm_fn=None)
        report = await auditor.audit(_valid_traversal())
        assert report.response_verdicts == []
        assert report.routing_leaks == []

    @pytest.mark.anyio
    async def test_llm_scores_responses(self) -> None:
        async def mock_llm(messages):
            return json.dumps({"score": 5, "routing_leak": False, "issues": []})

        auditor = SessionAuditor(flow=_simple_flow(), llm_fn=mock_llm)
        report = await auditor.audit(_valid_traversal())
        assert report.response_quality == 5.0
        assert len(report.response_verdicts) > 0


class TestLayer4SlotCompleteness:
    @pytest.mark.anyio
    async def test_captured_slot_reported(self) -> None:
        auditor = SessionAuditor(flow=_simple_flow())
        report = await auditor.audit(_valid_traversal())
        assert "name_given" in report.slot_coverage
        assert report.slot_coverage["name_given"] is True
        assert report.slot_completeness == 1.0

    @pytest.mark.anyio
    async def test_missing_slot_reported(self) -> None:
        traversal = _valid_traversal()
        traversal["traversal"][1]["criteria"]["criteria_map"] = {"name_given": False}
        auditor = SessionAuditor(flow=_simple_flow())
        report = await auditor.audit(traversal)
        assert report.slot_coverage["name_given"] is False
        assert report.slot_completeness == 0.0

    @pytest.mark.anyio
    async def test_no_criteria_gives_full_completeness(self) -> None:
        traversal = _valid_traversal()
        for step in traversal["traversal"]:
            step["criteria"] = None
        auditor = SessionAuditor(flow=_simple_flow())
        report = await auditor.audit(traversal)
        assert report.slot_completeness == 1.0


class TestFullAudit:
    @pytest.mark.anyio
    async def test_overall_score_computed(self) -> None:
        async def mock_llm(messages):
            content = messages[0].get("content", "")
            if "auditing a voice agent conversation" in content:
                return json.dumps({"correct": True, "confidence": "high", "preferred_edge": None, "reason": ""})
            return json.dumps({"score": 4, "routing_leak": False, "issues": []})

        auditor = SessionAuditor(flow=_simple_flow(), llm_fn=mock_llm)
        report = await auditor.audit(_valid_traversal())
        assert isinstance(report, AuditReport)
        assert 0.0 <= report.overall_score <= 1.0

    @pytest.mark.anyio
    async def test_to_markdown_contains_session_id(self) -> None:
        auditor = SessionAuditor(flow=_simple_flow())
        report = await auditor.audit(_valid_traversal())
        md = report.to_markdown()
        assert "test_001" in md

    @pytest.mark.anyio
    async def test_to_dict_serializable(self) -> None:
        auditor = SessionAuditor(flow=_simple_flow())
        report = await auditor.audit(_valid_traversal())
        d = report.to_dict()
        assert json.dumps(d)
        assert d["session_id"] == "test_001"