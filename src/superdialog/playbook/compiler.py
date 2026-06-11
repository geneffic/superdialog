"""Compile legacy ConversationFlow graphs into Playbooks (design doc §6)."""

from __future__ import annotations

import re
from typing import Any, Literal

from superdialog.flow.models import ConversationFlow, FlowNode
from superdialog.playbook.models import AdvanceRule, SlotSpec

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


# -- edge condition → AdvanceRule ---------------------------------------------

# Legacy flows spell equality as "==", "=", or prose "is".
_EQ = r"(?:==|=|\bis\b)"
_SUCCESS_RE = re.compile(rf"^\s*(\w+)\.success\s*{_EQ}\s*(true|false)\s*$", re.I)
_NOT_SUCCESS_RE = re.compile(r"^\s*not\s+(\w+)\.success\s*$", re.I)
_STATUS_RE = re.compile(rf"^\s*(\w+)\.status\s*{_EQ}\s*(\d{{3}})\s*$", re.I)
# An em-dash gloss that QUALIFIES the predicate must not be dropped.
_QUALIFIER_GLOSS_RE = re.compile(
    r"^(unless|but|only|if|when|except|and|or|while|until)\b", re.I
)


def _translate_predicate(text: str, store_keys: set[str]) -> str | None:
    """Translate one anchored data predicate to a runtime expr, or None."""
    if m := _SUCCESS_RE.match(text):
        key, value = m.group(1), m.group(2).lower()
        if key in store_keys:
            ok = f"results.{key}.ok"
            return ok if value == "true" else f"not {ok}"
    if m := _NOT_SUCCESS_RE.match(text):
        if m.group(1) in store_keys:
            return f"not results.{m.group(1)}.ok"
    if m := _STATUS_RE.match(text):
        if m.group(1) in store_keys:
            return f"results.{m.group(1)}.status == {m.group(2)}"
    return None


def compile_edge_condition(
    condition: str, store_keys: set[str], target: str
) -> AdvanceRule:
    """Compile a legacy edge condition into an AdvanceRule.

    Single-clause deterministic data predicates over known
    ``store_response_as`` keys become ``judge: expr`` rules:

    - ``X.success == true``  → ``results.X.ok``
    - ``X.success == false`` / ``not X.success`` → ``not results.X.ok``
    - ``X.status == NNN``    → ``results.X.status == NNN``

    Equality may be spelled ``==``, ``=``, or prose ``is`` (the legacy
    golf flow writes "X.success is true"). A trailing em-dash gloss
    ("X.success is false — route to retry") is stripped before matching
    unless it begins with a qualifier word (unless/but/only/...), which
    would change the predicate's meaning.

    Compound conditions ("A and B"), unknown keys, and anything else not
    confidently translatable stay ``judge: llm`` with the prose passed
    through verbatim — lossless beats clever; the Director can judge data
    conditions too, just slower.
    """
    expr = _translate_predicate(condition, store_keys)
    if expr is None:
        head, dash, gloss = condition.partition(" — ")
        if dash and not _QUALIFIER_GLOSS_RE.match(gloss.strip()):
            expr = _translate_predicate(head, store_keys)
    if expr is not None:
        return AdvanceRule(when=expr, judge="expr", to=target)
    return AdvanceRule(when=condition, judge="llm", to=target)


# -- edge input_schema → slot union -------------------------------------------

_JSON_TYPE: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "array",
    "object": "object",
}


def _slot_spec_from_property(prop: Any) -> SlotSpec:
    """Map one JSON-Schema property to an optional SlotSpec."""
    if not isinstance(prop, dict):
        return SlotSpec()
    enum = prop.get("enum")
    description = prop.get("description") or ""
    if isinstance(enum, list):
        return SlotSpec(
            type="enum", values=[str(v) for v in enum], description=description
        )
    json_type = _JSON_TYPE.get(prop.get("type", ""), "str")
    # The keys of _JSON_TYPE are exactly SlotSpec's literal members.
    return SlotSpec(type=json_type, description=description)  # type: ignore[arg-type]


def union_slot_schemas(
    node: FlowNode,
) -> tuple[dict[str, SlotSpec], dict[str, list[str]]]:
    """Union a node's edge ``input_schema`` properties into optional slots.

    Returns ``(slots, requires_by_edge)``:

    - ``slots``: every property declared by any edge schema, as an
      OPTIONAL ``SlotSpec`` (``required=False``) — per-branch requirements
      live on the rule, not the slot. On conflicting redeclarations the
      first declaration wins (consistent with ``Playbook.slot_spec``).
    - ``requires_by_edge``: edge id → that schema's ``required`` list,
      for every edge that has a (non-empty) schema.
    """
    slots: dict[str, SlotSpec] = {}
    requires_by_edge: dict[str, list[str]] = {}
    for edge in node.edges:
        schema = edge.input_schema
        if not isinstance(schema, dict) or not schema:
            continue
        properties = schema.get("properties") or {}
        for key, prop in properties.items():
            if key not in slots:
                slots[key] = _slot_spec_from_property(prop)
        requires_by_edge[edge.id] = list(schema.get("required") or [])
    return slots, requires_by_edge
