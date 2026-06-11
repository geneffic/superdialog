"""Compile legacy ConversationFlow graphs into Playbooks (design doc §6)."""

from __future__ import annotations

from typing import Literal

from superdialog.flow.models import ConversationFlow, FlowNode

NodeKind = Literal["conversational", "computational", "system"]


class FlowIndex:
    """Degree/classification index over a legacy flow graph.

    Builds an indegree map (counting both per-node edges and
    ``global_edges``) and a reverse-edge map, then classifies each node:

    - **system**: indegree 0 and not the initial node — only reachable
      via out-of-band triggers (webhooks, timers).
    - **computational**: routers or ``auto_proceed`` nodes — silent
      steps that never wait for the caller.
    - **conversational**: everything else — nodes that speak/listen.
    """

    def __init__(self, flow: ConversationFlow) -> None:
        self.flow = flow
        self._nodes: dict[str, FlowNode] = {n.id: n for n in flow.nodes}
        self.indegree: dict[str, int] = {n.id: 0 for n in flow.nodes}
        self.reverse_edges: dict[str, list[tuple[str, str]]] = {
            n.id: [] for n in flow.nodes
        }
        for node in flow.nodes:
            for edge in node.edges:
                target = edge.target_node_id
                if target in self.indegree:
                    self.indegree[target] += 1
                    self.reverse_edges[target].append((node.id, edge.id))
        for ge in flow.global_edges:
            if ge.target_node_id in self.indegree:
                self.indegree[ge.target_node_id] += 1

    def node(self, node_id: str) -> FlowNode:
        """Return the node with ``node_id`` (KeyError if absent)."""
        return self._nodes[node_id]

    def classify(self, node: FlowNode) -> NodeKind:
        """Classify a node as conversational, computational, or system."""
        if self.indegree.get(node.id, 0) == 0 and node.id != self.flow.initial_node:
            return "system"
        if node.node_type == "router" or node.auto_proceed:
            return "computational"
        return "conversational"
