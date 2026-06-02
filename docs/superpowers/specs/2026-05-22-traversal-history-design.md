# Traversal History — Design Spec

**Date:** 2026-05-22
**Status:** Approved

## Problem

After a chat session ends there is no record of which nodes were visited, what the bot said at each node, what the user said, or why transitions fired. Debugging requires re-running the session. Graph visualization of actual call paths is impossible.

## Goal

Capture every node visit, transition, bot message, user message, and criteria judge result during a `chat.py` session. Save to a structured JSON file. Provide a `view_traversal.py` script that renders the traversal as a PNG graph.

## Scope

- `chat.py` — incremental turn capture + post-session export
- `src/superdialog/Traversal_History/` — output directory for traversal JSON files
- `view_traversal.py` — standalone graph renderer (networkx + matplotlib)
- Zero changes to superdialog core (`machine.py`, adapters, models)

---

## Output File

**Location:** `src/superdialog/Traversal_History/traversal_<session_id>.json`

**session_id format:** `YYYYMMDD_HHMMSS` (from session start time)

---

## JSON Schema

```json
{
  "session_id": "20260522_143201",
  "flow_file": "generated_flow.json",
  "model": "openai/gpt-4.1",
  "started_at": "2026-05-22T14:32:01",
  "ended_at": "2026-05-22T14:38:45",
  "is_complete": true,

  "nodes": [
    {
      "id": "greeting",
      "name": "Greeting",
      "instruction": "Agent says: Namaste!..."
    }
  ],

  "traversal": [
    {
      "step": 1,
      "from_node": null,
      "to_node": "greeting",
      "edge_id": null,
      "timestamp": "2026-05-22T14:32:01",
      "node_instruction": "Agent says: Namaste!...",
      "bot_message": "Aap mujhe sun pa rahe hain?",
      "user_message": null,
      "criteria": null
    },
    {
      "step": 2,
      "from_node": "greeting",
      "to_node": "identity_verification",
      "edge_id": "greeting_to_identity_verification",
      "timestamp": "2026-05-22T14:32:18",
      "node_instruction": "Agent says: Aapki safety ke liye...",
      "bot_message": "Namaste! Main Isha...",
      "user_message": "haan krlo",
      "criteria": {
        "met": true,
        "recommended_edge_id": "greeting_to_identity_verification",
        "reason": "Caller confirmed availability",
        "slots": {}
      }
    }
  ],

  "graph": {
    "nodes": [
      {"id": "greeting", "name": "Greeting", "visited": true, "visit_count": 1, "is_final": false}
    ],
    "edges": [
      {
        "id": "greeting_to_identity_verification",
        "source": "greeting",
        "target": "identity_verification",
        "condition": "Caller agrees or is available and ready to proceed",
        "traversed": true,
        "traversed_at_step": 2
      }
    ]
  }
}
```

### Field reference

| Field | Source | Notes |
|-------|--------|-------|
| `traversal[].node_instruction` | `flow.nodes[id].instruction` | Looked up by `to_node` id |
| `traversal[].bot_message` | captured in `chat_turns` list during loop | Bot response for that step |
| `traversal[].user_message` | captured in `chat_turns` list during loop | User input that triggered transition |
| `traversal[].criteria` | `machine._machine.context.data.transition_log[i].criteria_met` | `None` for step 1 (initial node entry) |
| `graph.nodes[].visit_count` | `machine._machine.context.state.visit_count` | How many times node was entered |
| `graph.edges[].traversed` | match `edge_id` against `transition_log` | True if edge appears in log |

---

## Architecture

### In `chat.py`

**`chat_turns` list** — built incrementally in the loop:
```python
chat_turns: list[dict] = []
# After first turn (initial greeting):
chat_turns.append({"step": 1, "bot": first.text, "user": None, "node": node, "ts": now()})
# After each subsequent turn:
chat_turns.append({"step": n, "bot": turn.text, "user": raw, "node": node, "ts": now()})
```

**`build_traversal(machine, chat_turns, flow, source, model, started_at)`** — called once after loop exits:
- Pulls `transition_log` from `machine._machine.context.data.transition_log`
- Pulls `visit_count` from `machine._machine.context.state.visit_count`
- Aligns `chat_turns` with `transition_log` by step index
- Builds `nodes`, `traversal`, and `graph` sections
- Returns the full dict

**`save_traversal(traversal_dict, out_dir)`** — writes JSON:
```python
path = out_dir / f"traversal_{session_id}.json"
path.write_text(json.dumps(traversal_dict, indent=2, ensure_ascii=False))
```

`out_dir` = `Path(__file__).parent / "src/superdialog/Traversal_History"`

Directory is created on first write if missing.

---

## `view_traversal.py`

**Location:** next to `chat.py`

**Behavior:**
- No args → loads most recent file in `Traversal_History/`
- Arg = path → loads that file
- Draws directed graph with networkx + matplotlib
- Visited nodes: green; unvisited: grey; final nodes: red border
- Traversed edges: solid blue; untravelled edges: dashed grey
- Edge labels: edge id (shortened)
- Saves `graph_<session_id>.png` next to the traversal file
- Prints path to saved PNG

**Usage:**
```bash
python view_traversal.py                          # latest session
python view_traversal.py traversal_20260522.json  # specific session
```

**Dependencies:** `networkx`, `matplotlib` (both pip-installable, not in core superdialog deps)

---

## Error Handling

- If `machine._machine` is `None` (session never started): save partial traversal with what exists in `chat_turns`
- If `Traversal_History/` doesn't exist: create it
- If `view_traversal.py` can't find networkx: print install instruction and exit cleanly

---

## Files Changed

| File | Change |
|------|--------|
| `chat.py` | Add `chat_turns` capture, `build_traversal()`, `save_traversal()`, call at session end |
| `view_traversal.py` | New file — graph renderer |
| `src/superdialog/Traversal_History/` | New directory — traversal output |
| `src/superdialog/Traversal_History/.gitkeep` | Keep dir in git, ignore JSON files |

No changes to superdialog core.