"""E2E tests that load real flow JSONs through the SDK runner."""

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

from superdialog.machine.runner import run_flow  # noqa: E402

TEMP_DIR = Path(__file__).resolve().parents[4] / "temp"


def _make_sequenced_llm(edge_sequence: list[str]):
    """Create a mock LLM that returns edges in sequence for criteria evaluation."""
    idx = {"i": 0}

    async def llm(messages: list[dict]) -> str:
        sys_content = messages[0].get("content", "")
        if "evaluating" in sys_content:
            eid = edge_sequence[idx["i"]] if idx["i"] < len(edge_sequence) else None
            idx["i"] += 1
            return json.dumps(
                {
                    "all_required_met": True,
                    "recommended_edge_id": eid,
                    "reason": "scripted",
                }
            )
        return "SDK test reply"

    return llm


def _extract_happy_path(flow_path: str) -> list[str]:
    """Extract a happy path by following the first edge of each non-final node."""
    flow_data = json.loads(Path(flow_path).read_text())
    nodes = {n["id"]: n for n in flow_data.get("nodes", [])}
    edge_seq: list[str] = []
    current = flow_data.get("initial_node", "")
    visited: set[str] = set()
    while current and current not in visited:
        visited.add(current)
        node = nodes.get(current)
        if not node or node.get("is_final"):
            break
        edges = node.get("edges", [])
        if not edges:
            break
        edge = edges[0]
        edge_seq.append(edge["id"])
        current = edge.get("target_node_id", "")
    return edge_seq


@pytest.mark.skipif(
    not (TEMP_DIR / "tech_support_lite_flow.json").exists(),
    reason="Flow JSON not present",
)
class TestSDKWithTechSupport:
    @pytest.mark.anyio
    async def test_happy_path_reaches_final(self):
        flow_path = str(TEMP_DIR / "tech_support_lite_flow.json")
        edge_seq = _extract_happy_path(flow_path)
        llm = _make_sequenced_llm(edge_seq)
        user_msgs = [f"input {i}" for i in range(len(edge_seq))]
        result = await run_flow(
            flow=flow_path,
            llm_fn=llm,
            user_messages=user_msgs,
        )
        assert result.is_complete or len(result.transitions) == len(edge_seq)
        assert len(result.transitions) >= 1


@pytest.mark.skipif(
    not (TEMP_DIR / "kairali_lite_flow.json").exists(),
    reason="Flow JSON not present",
)
class TestSDKWithKairaliLite:
    @pytest.mark.anyio
    async def test_happy_path_traversal(self):
        flow_path = str(TEMP_DIR / "kairali_lite_flow.json")
        edge_seq = _extract_happy_path(flow_path)
        llm = _make_sequenced_llm(edge_seq)
        user_msgs = [f"input {i}" for i in range(len(edge_seq))]
        result = await run_flow(
            flow=flow_path,
            llm_fn=llm,
            user_messages=user_msgs,
        )
        assert len(result.transitions) >= 1


@pytest.mark.skipif(
    not (TEMP_DIR / "outbound_sales_lite_flow.json").exists(),
    reason="Flow JSON not present",
)
class TestSDKWithOutboundSales:
    @pytest.mark.anyio
    async def test_happy_path_traversal(self):
        flow_path = str(TEMP_DIR / "outbound_sales_lite_flow.json")
        edge_seq = _extract_happy_path(flow_path)
        llm = _make_sequenced_llm(edge_seq)
        user_msgs = [f"input {i}" for i in range(len(edge_seq))]
        result = await run_flow(
            flow=flow_path,
            llm_fn=llm,
            user_messages=user_msgs,
        )
        assert len(result.transitions) >= 1


@pytest.mark.skipif(
    not (TEMP_DIR / "survey_feedback_lite_flow.json").exists(),
    reason="Flow JSON not present",
)
class TestSDKWithSurvey:
    @pytest.mark.anyio
    async def test_happy_path_traversal(self):
        flow_path = str(TEMP_DIR / "survey_feedback_lite_flow.json")
        edge_seq = _extract_happy_path(flow_path)
        llm = _make_sequenced_llm(edge_seq)
        user_msgs = [f"input {i}" for i in range(len(edge_seq))]
        result = await run_flow(
            flow=flow_path,
            llm_fn=llm,
            user_messages=user_msgs,
        )
        assert len(result.transitions) >= 1
