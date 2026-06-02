"""Tier 1: Structural graph analysis tests for real flows.

Tests FlowGraphAnalyzer against vajiram and kairali flow JSONs.
No LLM calls — pure deterministic validation.
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
from superdialog.machine.eval.graph_analyzer import FlowGraphAnalyzer  # noqa: E402

TEMP_DIR = Path(__file__).resolve().parents[4] / "temp"


def _load_flow(filename: str) -> ConversationFlow:
    path = TEMP_DIR / filename
    raw = json.loads(path.read_text())
    return ConversationFlow(**raw)


@pytest.fixture
def vajiram_flow() -> ConversationFlow:
    return _load_flow("vajiram_saanvi_inbound_flow.json")


@pytest.fixture
def kairali_flow() -> ConversationFlow:
    return _load_flow("kairali_flow 2.json")


# ------------------------------------------------------------------
# Vajiram flow structural tests
# ------------------------------------------------------------------


class TestVajiramStructure:
    def test_valid_structure(self, vajiram_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        errors = analyzer.validate_structure()
        assert errors == [], f"Structural errors: {errors}"

    def test_node_count(self, vajiram_flow: ConversationFlow) -> None:
        assert len(vajiram_flow.nodes) == 17

    def test_initial_node_exists(self, vajiram_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        assert vajiram_flow.initial_node == "opening"
        node_ids = {n.id for n in vajiram_flow.nodes}
        assert "opening" in node_ids

    def test_single_final_node(self, vajiram_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        finals = analyzer.all_final_nodes()
        assert finals == {"closing"}

    def test_edge_count(self, vajiram_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        assert len(analyzer.all_edge_ids()) == 32

    def test_no_orphan_nodes(self, vajiram_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        assert analyzer.orphan_nodes() == set()

    def test_all_nodes_reachable(self, vajiram_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        reachable = analyzer.reachable_nodes()
        all_ids = {n.id for n in vajiram_flow.nodes}
        assert reachable == all_ids

    def test_no_dangling_edges(self, vajiram_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        assert analyzer.dangling_edges() == []

    def test_final_node_reachable(self, vajiram_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        assert analyzer.unreachable_finals() == set()

    def test_paths_exist_to_final(self, vajiram_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        paths = analyzer.enumerate_paths()
        assert len(paths) > 0
        for path in paths:
            last_target = path[-1][1]
            assert last_target == "closing"

    def test_edge_coverage_from_all_paths(self, vajiram_flow: ConversationFlow) -> None:
        """All edges should be covered except back-edges that loop
        to an earlier node without reaching a final node first
        (e.g. callback_correction loops back to callback_collect_name).
        """
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        paths = analyzer.enumerate_paths()
        covered, uncovered = analyzer.edge_coverage_from_paths(paths)
        # callback_correction is a correction loop that only revisits
        # already-traversed nodes — DFS with max_cycle_visits=1 covers it
        paths_2 = analyzer.enumerate_paths(max_cycle_visits=2)
        covered_2, uncovered_2 = analyzer.edge_coverage_from_paths(paths_2)
        assert (
            len(uncovered_2) == 0
        ), f"Uncovered edges with cycle visit 2: {uncovered_2}"

    def test_path_to_specific_nodes(self, vajiram_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        path = analyzer.path_to_node("callback_collect_name")
        assert path is not None
        assert len(path) >= 3

        path_to_closing = analyzer.path_to_node("closing")
        assert path_to_closing is not None

    def test_path_to_initial_is_empty(self, vajiram_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        path = analyzer.path_to_node("opening")
        assert path == []

    def test_self_loop_edges(self, vajiram_flow: ConversationFlow) -> None:
        """existing_faq_answer and external_faq_answer have self-loops."""
        analyzer = FlowGraphAnalyzer(vajiram_flow)
        edges = analyzer.all_edges()
        self_loops = [(nid, eid, tid) for nid, eid, tid in edges if nid == tid]
        loop_ids = {eid for _, eid, _ in self_loops}
        assert "existing_followup_faq" in loop_ids
        assert "external_followup_faq" in loop_ids


# ------------------------------------------------------------------
# Kairali flow structural tests
# ------------------------------------------------------------------


class TestKairaliStructure:
    def test_valid_structure(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        errors = analyzer.validate_structure()
        assert errors == [], f"Structural errors: {errors}"

    def test_node_count(self, kairali_flow: ConversationFlow) -> None:
        assert len(kairali_flow.nodes) == 45

    def test_initial_node(self, kairali_flow: ConversationFlow) -> None:
        assert kairali_flow.initial_node == "greeting"

    def test_final_nodes(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        finals = analyzer.all_final_nodes()
        expected = {
            "person_not_available_confirm",
            "already_spoke_close",
            "nobody_enquired_close",
            "reschedule_confirm",
            "not_interested_close",
            "unqualified_close",
            "closing",
        }
        assert finals == expected

    def test_edge_count(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        assert len(analyzer.all_edge_ids()) == 65

    def test_no_orphan_nodes(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        assert analyzer.orphan_nodes() == set()

    def test_all_nodes_reachable(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        reachable = analyzer.reachable_nodes()
        all_ids = {n.id for n in kairali_flow.nodes}
        assert reachable == all_ids

    def test_no_dangling_edges(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        assert analyzer.dangling_edges() == []

    def test_all_finals_reachable(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        assert analyzer.unreachable_finals() == set()

    def test_paths_reach_final_nodes(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        paths = analyzer.enumerate_paths()
        assert len(paths) > 0
        finals = analyzer.all_final_nodes()
        for path in paths:
            last_target = path[-1][1]
            assert last_target in finals

    def test_edge_coverage(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        paths = analyzer.enumerate_paths()
        covered, uncovered = analyzer.edge_coverage_from_paths(paths)
        assert len(uncovered) == 0, f"Uncovered edges: {uncovered}"

    def test_enquiry_open_has_10_edges(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        edges = analyzer.edges_for_node("enquiry_open")
        assert len(edges) == 10

    def test_path_to_contact_capture(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        path = analyzer.path_to_node("contact_capture")
        assert path is not None
        assert len(path) >= 3

    def test_greeting_has_five_branches(self, kairali_flow: ConversationFlow) -> None:
        analyzer = FlowGraphAnalyzer(kairali_flow)
        edges = analyzer.edges_for_node("greeting")
        assert len(edges) == 5


# ------------------------------------------------------------------
# FlowGraphAnalyzer unit tests (synthetic flows)
# ------------------------------------------------------------------


def _linear_flow() -> ConversationFlow:
    """A -> B -> C (final)."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="a",
        nodes=[
            FlowNode(
                id="a",
                name="A",
                instruction="go",
                edges=[Edge(id="a_b", condition="next", target_node_id="b")],
            ),
            FlowNode(
                id="b",
                name="B",
                instruction="go",
                edges=[Edge(id="b_c", condition="next", target_node_id="c")],
            ),
            FlowNode(id="c", name="C", instruction="done", is_final=True),
        ],
    )


def _branching_flow() -> ConversationFlow:
    """A -> B or C (both final)."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="a",
        nodes=[
            FlowNode(
                id="a",
                name="A",
                instruction="pick",
                edges=[
                    Edge(id="a_b", condition="left", target_node_id="b"),
                    Edge(id="a_c", condition="right", target_node_id="c"),
                ],
            ),
            FlowNode(id="b", name="B", instruction="done", is_final=True),
            FlowNode(id="c", name="C", instruction="done", is_final=True),
        ],
    )


def _flow_with_orphan() -> ConversationFlow:
    """A -> B (final), orphan node D."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="a",
        nodes=[
            FlowNode(
                id="a",
                name="A",
                instruction="go",
                edges=[Edge(id="a_b", condition="next", target_node_id="b")],
            ),
            FlowNode(id="b", name="B", instruction="done", is_final=True),
            FlowNode(
                id="d",
                name="D",
                instruction="orphan",
                edges=[Edge(id="d_b", condition="x", target_node_id="b")],
            ),
        ],
    )


def _flow_with_dangling() -> ConversationFlow:
    """A has edge to non-existent node Z."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="a",
        nodes=[
            FlowNode(
                id="a",
                name="A",
                instruction="go",
                edges=[
                    Edge(id="a_z", condition="bad", target_node_id="z"),
                    Edge(id="a_b", condition="ok", target_node_id="b"),
                ],
            ),
            FlowNode(id="b", name="B", instruction="done", is_final=True),
        ],
    )


def _flow_with_cycle() -> ConversationFlow:
    """A -> B -> A (cycle) or B -> C (final)."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="a",
        nodes=[
            FlowNode(
                id="a",
                name="A",
                instruction="go",
                edges=[Edge(id="a_b", condition="next", target_node_id="b")],
            ),
            FlowNode(
                id="b",
                name="B",
                instruction="pick",
                edges=[
                    Edge(id="b_a", condition="back", target_node_id="a"),
                    Edge(id="b_c", condition="done", target_node_id="c"),
                ],
            ),
            FlowNode(id="c", name="C", instruction="done", is_final=True),
        ],
    )


class TestAnalyzerLinearFlow:
    def test_all_edges(self) -> None:
        analyzer = FlowGraphAnalyzer(_linear_flow())
        edges = analyzer.all_edges()
        assert len(edges) == 2
        assert ("a", "a_b", "b") in edges
        assert ("b", "b_c", "c") in edges

    def test_reachable(self) -> None:
        analyzer = FlowGraphAnalyzer(_linear_flow())
        assert analyzer.reachable_nodes() == {"a", "b", "c"}

    def test_single_path(self) -> None:
        analyzer = FlowGraphAnalyzer(_linear_flow())
        paths = analyzer.enumerate_paths()
        assert len(paths) == 1
        assert paths[0] == [("a_b", "b"), ("b_c", "c")]

    def test_full_coverage(self) -> None:
        analyzer = FlowGraphAnalyzer(_linear_flow())
        paths = analyzer.enumerate_paths()
        covered, uncovered = analyzer.edge_coverage_from_paths(paths)
        assert uncovered == set()

    def test_validate_clean(self) -> None:
        analyzer = FlowGraphAnalyzer(_linear_flow())
        assert analyzer.validate_structure() == []


class TestAnalyzerBranchingFlow:
    def test_two_paths(self) -> None:
        analyzer = FlowGraphAnalyzer(_branching_flow())
        paths = analyzer.enumerate_paths()
        assert len(paths) == 2

    def test_both_finals_reached(self) -> None:
        analyzer = FlowGraphAnalyzer(_branching_flow())
        paths = analyzer.enumerate_paths()
        targets = {path[-1][1] for path in paths}
        assert targets == {"b", "c"}


class TestAnalyzerOrphanDetection:
    def test_orphan_found(self) -> None:
        analyzer = FlowGraphAnalyzer(_flow_with_orphan())
        assert analyzer.orphan_nodes() == {"d"}

    def test_validation_reports_orphan(self) -> None:
        analyzer = FlowGraphAnalyzer(_flow_with_orphan())
        errors = analyzer.validate_structure()
        assert any("'d'" in e and "unreachable" in e for e in errors)


class TestAnalyzerDanglingEdge:
    def test_dangling_found(self) -> None:
        analyzer = FlowGraphAnalyzer(_flow_with_dangling())
        dangling = analyzer.dangling_edges()
        assert len(dangling) == 1
        assert dangling[0] == ("a", "a_z", "z")

    def test_validation_reports_dangling(self) -> None:
        analyzer = FlowGraphAnalyzer(_flow_with_dangling())
        errors = analyzer.validate_structure()
        assert any("a_z" in e and "non-existent" in e for e in errors)


def _flow_with_global_edges() -> ConversationFlow:
    """A -> B -> C (final), with 2 global edges."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="a",
        nodes=[
            FlowNode(
                id="a",
                name="A",
                instruction="go",
                edges=[
                    Edge(id="a_b", condition="next", target_node_id="b"),
                ],
            ),
            FlowNode(
                id="b",
                name="B",
                instruction="go",
                edges=[
                    Edge(id="b_c", condition="next", target_node_id="c"),
                ],
            ),
            FlowNode(id="c", name="C", instruction="done", is_final=True),
            FlowNode(
                id="help",
                name="Help",
                instruction="help",
                is_final=True,
            ),
        ],
        global_edges=[
            Edge(
                id="global_help",
                condition="user asks for help",
                target_node_id="help",
            ),
            Edge(
                id="global_hangup",
                condition="user hangs up",
                target_node_id="c",
            ),
        ],
    )


class TestAnalyzerGlobalEdges:
    def test_all_global_edges(self) -> None:
        analyzer = FlowGraphAnalyzer(_flow_with_global_edges())
        global_edges = analyzer.all_global_edges()
        assert len(global_edges) == 2
        assert ("global", "global_help", "help") in global_edges
        assert ("global", "global_hangup", "c") in global_edges

    def test_all_edges_includes_global(self) -> None:
        analyzer = FlowGraphAnalyzer(_flow_with_global_edges())
        all_edges = analyzer.all_edges()
        # 2 node edges + 2 global edges
        assert len(all_edges) == 4
        ids = {e[1] for e in all_edges}
        assert "global_help" in ids
        assert "global_hangup" in ids

    def test_all_edge_ids_includes_global(self) -> None:
        analyzer = FlowGraphAnalyzer(_flow_with_global_edges())
        ids = analyzer.all_edge_ids()
        assert "global_help" in ids
        assert "global_hangup" in ids
        assert len(ids) == 4

    def test_coverage_includes_global_edges(self) -> None:
        analyzer = FlowGraphAnalyzer(_flow_with_global_edges())
        paths = analyzer.enumerate_paths()
        _, uncovered = analyzer.edge_coverage_from_paths(paths)
        # Global edges are not traversed by DFS paths
        assert "global_help" in uncovered
        assert "global_hangup" in uncovered

    def test_validate_global_edge_dangling(self) -> None:
        flow = ConversationFlow(
            system_prompt="test",
            initial_node="a",
            nodes=[
                FlowNode(
                    id="a",
                    name="A",
                    instruction="go",
                    is_final=True,
                ),
            ],
            global_edges=[
                Edge(
                    id="g_bad",
                    condition="x",
                    target_node_id="nonexistent",
                ),
            ],
        )
        analyzer = FlowGraphAnalyzer(flow)
        errors = analyzer.validate_structure()
        assert any("g_bad" in e and "non-existent" in e for e in errors)

    def test_no_global_edges_is_fine(self) -> None:
        analyzer = FlowGraphAnalyzer(_linear_flow())
        assert analyzer.all_global_edges() == []
        assert len(analyzer.all_edge_ids()) == 2


class TestAnalyzerCycles:
    def test_cycle_bounded_paths(self) -> None:
        analyzer = FlowGraphAnalyzer(_flow_with_cycle())
        paths = analyzer.enumerate_paths(max_cycle_visits=1)
        assert len(paths) >= 1
        for path in paths:
            assert path[-1][1] == "c"

    def test_cycle_with_higher_bound(self) -> None:
        analyzer = FlowGraphAnalyzer(_flow_with_cycle())
        paths_1 = analyzer.enumerate_paths(max_cycle_visits=1)
        paths_2 = analyzer.enumerate_paths(max_cycle_visits=2)
        assert len(paths_2) >= len(paths_1)

    def test_all_edges_covered_with_cycle(self) -> None:
        analyzer = FlowGraphAnalyzer(_flow_with_cycle())
        paths = analyzer.enumerate_paths(max_cycle_visits=2)
        covered, uncovered = analyzer.edge_coverage_from_paths(paths)
        assert uncovered == set()
