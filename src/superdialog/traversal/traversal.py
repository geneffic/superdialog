"""Traversal history builder — records a full dialog session to JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superdialog.dialog_machine import DialogMachine
    from superdialog.flow.models import ConversationFlow


def build_traversal(
    machine: "DialogMachine",
    chat_turns: list[dict[str, Any]],
    flow: "ConversationFlow",
    source: str,
    model: str,
    started_at: datetime,
) -> dict[str, Any]:
    """Build traversal JSON from a completed chat session.

    Args:
        machine: The DialogMachine instance after the session.
        chat_turns: List of dicts with keys: step, bot, user, node, ts.
        flow: The ConversationFlow used in the session.
        source: Display name for the flow file (e.g. "flow.json").
        model: Model URI used (e.g. "openai/gpt-4.1-mini").
        started_at: UTC datetime when the session started.
    """
    ended_at = datetime.now(timezone.utc)
    session_id = started_at.strftime("%Y%m%d_%H%M%S_%f")[:20]

    # Node lookup: id -> {id, name, instruction, is_final}
    node_lookup: dict[str, dict[str, Any]] = {
        n.id: {
            "id": n.id,
            "name": n.name,
            "instruction": n.instruction or (n.static_text if hasattr(n, "static_text") else "") or "",
            "is_final": n.is_final,
        }
        for n in flow.nodes
    }

    # All edges from flow (node edges + global edges)
    all_edges: list[dict[str, Any]] = []
    for n in flow.nodes:
        for e in n.edges:
            all_edges.append({
                "id": e.id,
                "source": n.id,
                "target": e.target_node_id,
                "condition": e.condition or "",
            })
    for e in getattr(flow, "global_edges", []):
        all_edges.append({
            "id": e.id,
            "source": "__global__",
            "target": e.target_node_id,
            "condition": e.condition or "",
        })

    # Pull state from machine internals
    # Use backward-compat property names (ctx.transition_log, ctx.visit_count,
    # ctx.action_log) which work on both real FlowContext and test mocks.
    transition_log: list[Any] = []
    visit_count: dict[str, int] = {}
    action_log: list[Any] = []
    is_complete = False
    if machine._machine is not None:
        ctx = machine._machine.context
        transition_log = list(ctx.transition_log)
        visit_count = dict(ctx.visit_count)
        action_log = list(ctx.action_log)
        is_complete = machine._machine.is_complete

    # Build action lookup: node_id -> list of ActionRecord dicts
    actions_by_node: dict[str, list[dict[str, Any]]] = {}
    for rec in action_log:
        entry = {
            "action_id": rec.action_id,
            "trigger": rec.trigger,
            "url": rec.url,
            "method": rec.method,
            "status": rec.status,
            "success": rec.success,
            "result_data": rec.result_data,
        }
        actions_by_node.setdefault(rec.node_id, []).append(entry)

    # Build traversal steps
    traversal_steps: list[dict[str, Any]] = []

    if chat_turns:
        first_turn = chat_turns[0]
        first_node = first_turn.get("node", "")
        traversal_steps.append({
            "step": 1,
            "from_node": None,
            "to_node": first_node,
            "edge_id": None,
            "timestamp": first_turn.get("ts", ""),
            "node_instruction": node_lookup.get(first_node, {}).get("instruction", ""),
            "bot_message": first_turn.get("bot", ""),
            "user_message": first_turn.get("user"),
            "criteria": None,
            "actions": actions_by_node.get(first_node, []),
        })

    # Decide message sourcing ONCE for the whole log. If any record carries
    # attribution, this is a "new" log -> trust records entirely (robust to
    # router chaining; a router hop legitimately has bot_message=""). Only a
    # fully-legacy log (no record carries any message) falls back to the old
    # positional chat_turns pairing. This avoids re-introducing the mis-pairing
    # for router hops whose genuine bot_message is "".
    log_has_attribution = any(
        (getattr(rec, "user_message", None) is not None)
        or (getattr(rec, "bot_message", "") or "")
        for rec in transition_log
    )

    for i, rec in enumerate(transition_log):
        turn = chat_turns[i + 1] if i + 1 < len(chat_turns) else {}
        if log_has_attribution:
            bot_message = getattr(rec, "bot_message", "") or ""
            user_message = getattr(rec, "user_message", None)
        else:
            bot_message = turn.get("bot")
            user_message = turn.get("user")
        # met is True when criteria all pass OR there were no criteria to gate on
        # (an empty criteria_met dict means "nothing required", not "failed").
        met = (not rec.skipped) and (
            all(rec.criteria_met.values()) if rec.criteria_met else True
        )
        criteria = {
            "met": met,
            "skipped": rec.skipped,
            "edge_id": rec.edge_id,
            "criteria_map": dict(rec.criteria_met),
        }
        traversal_steps.append({
            "step": i + 2,
            "from_node": rec.from_node,
            "to_node": rec.to_node,
            "edge_id": rec.edge_id,
            "timestamp": datetime.fromtimestamp(
                rec.timestamp, tz=timezone.utc
            ).isoformat(),
            "node_instruction": node_lookup.get(rec.to_node, {}).get(
                "instruction", ""
            ),
            "bot_message": bot_message,
            "user_message": user_message,
            "criteria": criteria,
            "actions": actions_by_node.get(rec.to_node, []),
        })

    # Graph annotations
    traversed_edge_ids = {rec.edge_id for rec in transition_log}
    edge_step_map = {rec.edge_id: (i + 2) for i, rec in enumerate(transition_log)}

    # Synthesize virtual edges from transition_log for any edge_id not
    # already present in all_edges (e.g. when flow nodes have no edges defined).
    existing_edge_ids = {e["id"] for e in all_edges}
    for rec in transition_log:
        if rec.edge_id not in existing_edge_ids:
            all_edges.append({
                "id": rec.edge_id,
                "source": rec.from_node,
                "target": rec.to_node,
                "condition": "",
            })
            existing_edge_ids.add(rec.edge_id)

    graph_nodes = [
        {
            "id": n.id,
            "name": n.name,
            "visited": n.id in visit_count,
            "visit_count": visit_count.get(n.id, 0),
            "is_final": n.is_final,
        }
        for n in flow.nodes
    ]
    graph_edges = [
        {
            "id": e["id"],
            "source": e["source"],
            "target": e["target"],
            "condition": e["condition"],
            "traversed": e["id"] in traversed_edge_ids,
            "traversed_at_step": edge_step_map.get(e["id"]),
        }
        for e in all_edges
    ]

    return {
        "session_id": session_id,
        "flow_file": source,
        "model": model,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "is_complete": is_complete,
        "nodes": list(node_lookup.values()),
        "traversal": traversal_steps,
        "graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
        },
    }


def save_traversal(traversal: dict[str, Any], out_dir: str | Path) -> Path:
    """Write traversal JSON to out_dir. Creates dir if missing. Returns path written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = traversal["session_id"]
    path = out_dir / f"traversal_{session_id}.json"
    path.write_text(json.dumps(traversal, indent=2, ensure_ascii=False))
    return path
