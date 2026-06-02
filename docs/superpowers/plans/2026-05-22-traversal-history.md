# Traversal History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture every node visit, transition, and message during a `chat.py` session and save a structured JSON file; provide `view_traversal.py` to render it as a PNG graph.

**Architecture:** All capture logic lives in `chat.py` — zero core changes. A `chat_turns` list collects bot/user messages per turn; after the loop exits, `build_traversal()` merges that with `machine._machine.context.data.transition_log` and `visit_count` into a single JSON. `view_traversal.py` reads that JSON and renders a directed graph via networkx + matplotlib.

**Tech Stack:** Python 3.10+, `json` (stdlib), `datetime` (stdlib), `pathlib` (stdlib), `networkx`, `matplotlib`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `chat.py` | Modify | Add `chat_turns` capture, `build_traversal()`, `save_traversal()`, wire into `chat()` |
| `view_traversal.py` | Create | Standalone graph renderer — reads traversal JSON, writes PNG |
| `src/superdialog/Traversal_History/.gitkeep` | Create | Keep directory in git; JSON files are gitignored |

---

## Task 1: Create output directory

**Files:**
- Create: `src/superdialog/Traversal_History/.gitkeep`

- [ ] **Step 1: Create directory and .gitkeep**

```bash
mkdir -p /home/ankit/Unpod/super-sanyam/super/superdialog/src/superdialog/traversal
touch /home/ankit/Unpod/super-sanyam/super/superdialog/src/superdialog/traversal/.gitkeep
```

- [ ] **Step 2: Verify directory exists**

```bash
ls /home/ankit/Unpod/super-sanyam/super/superdialog/src/superdialog/traversal/
```

Expected: `.gitkeep`

---

## Task 2: Add `build_traversal()` and `save_traversal()` to `chat.py`

**Files:**
- Modify: `chat.py`

These two functions go after the `print_node_status()` function and before `generate_and_save()`.

> **Note on `criteria`:** `TransitionRecord` stores `criteria_met: dict[str, bool]` (e.g. `{"node_spoken": True, "user_spoke": True}`) and `edge_id`. The `reason` text from `CriteriaResult` is ephemeral and not persisted in context, so it is not included.

- [ ] **Step 1: Add imports at top of `chat.py`**

Open `chat.py`. The existing imports are:
```python
from __future__ import annotations
import asyncio, os, sys, textwrap, argparse
from dotenv import load_dotenv
```

Change to:
```python
from __future__ import annotations
import asyncio, os, sys, textwrap, argparse, json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
```

- [ ] **Step 2: Add `TRAVERSAL_DIR` constant after `SYSTEM_PROMPT_PATH`**

Current `chat.py` has:
```python
FLOW_PATH        = "/home/ankit/Unpod/super-sanyam/super/superdialog/generated_flow.json"
SYSTEM_PROMPT_PATH = "/home/ankit/Unpod/super-sanyam/super/superdialog/generated_system_prompt.txt"
```

Add after `SYSTEM_PROMPT_PATH`:
```python
TRAVERSAL_DIR = Path(__file__).parent / "src/superdialog/traversal"
```

- [ ] **Step 3: Add `build_traversal()` function after `print_node_status()`**

Insert this function after `print_node_status()` and before `generate_and_save()`:

```python
def build_traversal(
    machine,
    chat_turns: list[dict],
    flow,
    source: str,
    model: str,
    started_at: datetime,
) -> dict:
    """Build traversal JSON from completed chat session."""
    ended_at = datetime.now(timezone.utc)
    session_id = started_at.strftime("%Y%m%d_%H%M%S")

    # Node lookup: id -> {id, name, instruction}
    node_lookup = {
        n.id: {
            "id": n.id,
            "name": n.name,
            "instruction": n.instruction or n.static_text or "",
            "is_final": n.is_final,
        }
        for n in flow.nodes
    }

    # All edges from flow
    all_edges = []
    for n in flow.nodes:
        for e in n.edges:
            all_edges.append({
                "id": e.id,
                "source": n.id,
                "target": e.target_node_id,
                "condition": e.condition or "",
            })

    # Transition log from machine context (may be None if session never started)
    transition_log = []
    visit_count: dict = {}
    is_complete = False
    if machine._machine is not None:
        ctx = machine._machine.context
        transition_log = list(ctx.data.transition_log)
        visit_count = dict(ctx.state.visit_count)
        is_complete = machine._machine.is_complete

    # Build traversal steps
    # Step 1 = initial node entry (no transition record)
    # Steps 2+ = one per TransitionRecord
    traversal_steps = []

    if chat_turns:
        first_turn = chat_turns[0]
        traversal_steps.append({
            "step": 1,
            "from_node": None,
            "to_node": first_turn["node"],
            "edge_id": None,
            "timestamp": first_turn["ts"],
            "node_instruction": node_lookup.get(first_turn["node"], {}).get("instruction", ""),
            "bot_message": first_turn["bot"],
            "user_message": None,
            "criteria": None,
        })

    for i, rec in enumerate(transition_log):
        turn = chat_turns[i + 1] if i + 1 < len(chat_turns) else {}
        criteria = {
            "met": all(rec.criteria_met.values()) if rec.criteria_met else True,
            "edge_id": rec.edge_id,
            "criteria_map": dict(rec.criteria_met),
        }
        traversal_steps.append({
            "step": i + 2,
            "from_node": rec.from_node,
            "to_node": rec.to_node,
            "edge_id": rec.edge_id,
            "timestamp": datetime.fromtimestamp(rec.timestamp, tz=timezone.utc).isoformat(),
            "node_instruction": node_lookup.get(rec.to_node, {}).get("instruction", ""),
            "bot_message": turn.get("bot"),
            "user_message": turn.get("user"),
            "criteria": criteria,
        })

    # Traversed edge ids
    traversed_edge_ids = {rec.edge_id for rec in transition_log}
    # Step → traversed_at lookup
    edge_step_map = {
        rec.edge_id: (i + 2) for i, rec in enumerate(transition_log)
    }

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


def save_traversal(traversal: dict, out_dir: Path) -> Path:
    """Write traversal JSON to out_dir. Creates dir if missing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = traversal["session_id"]
    path = out_dir / f"traversal_{session_id}.json"
    path.write_text(json.dumps(traversal, indent=2, ensure_ascii=False))
    return path
```

- [ ] **Step 4: Verify `chat.py` has no syntax errors**

```bash
cd /home/ankit/Unpod/super-sanyam/super/superdialog && python -c "import ast; ast.parse(open('chat.py').read()); print('OK')"
```

Expected: `OK`

---

## Task 3: Wire `chat_turns` capture into `chat()` loop

**Files:**
- Modify: `chat.py` — `chat()` function

- [ ] **Step 1: Add `chat_turns` list and `started_at` before the first turn**

In `chat()`, locate this block:
```python
    flow    = Flow.load(flow_path)
    machine = DialogMachine(flow=flow, llm=MODEL)
    source  = os.path.basename(flow_path)
```

Add two lines after it:
```python
    flow    = Flow.load(flow_path)
    machine = DialogMachine(flow=flow, llm=MODEL)
    source  = os.path.basename(flow_path)
    chat_turns: list[dict] = []
    started_at = datetime.now(timezone.utc)
```

- [ ] **Step 2: Capture first turn (initial greeting)**

Locate this block in `chat()`:
```python
    node = machine.state["node_id"]
    if first.text:
        msg = {"role": "assistant", "content": first.text, "_node": node}
        print_msg(msg)
```

Replace with:
```python
    node = machine.state["node_id"]
    if first.text:
        msg = {"role": "assistant", "content": first.text, "_node": node}
        print_msg(msg)
    chat_turns.append({
        "step": 1,
        "bot": first.text or "",
        "user": None,
        "node": node,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
```

- [ ] **Step 3: Capture each subsequent turn**

Locate this block at the bottom of the `while True` loop in `chat()`:
```python
        node = machine.state["node_id"]
        if turn.text:
            print_msg({"role": "assistant", "content": turn.text, "_node": node})
```

Replace with:
```python
        node = machine.state["node_id"]
        if turn.text:
            print_msg({"role": "assistant", "content": turn.text, "_node": node})
        chat_turns.append({
            "step": len(chat_turns) + 1,
            "bot": turn.text or "",
            "user": raw,
            "node": node,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
```

---

## Task 4: Call `build_traversal` + `save_traversal` at session end

**Files:**
- Modify: `chat.py` — end of `chat()` function

The `chat()` function currently ends right after the `while True` loop (after the `break` or `KeyboardInterrupt`). Add the save block after the loop.

- [ ] **Step 1: Add save call after the `while True` loop**

Locate the end of `chat()` — after the `while True:` block ends (after the last `break`). Add:

```python
    # Save traversal after loop exits
    if chat_turns:
        try:
            flow_obj = flow  # already loaded above
            traversal = build_traversal(
                machine, chat_turns, flow_obj, source, MODEL, started_at
            )
            saved_path = save_traversal(traversal, TRAVERSAL_DIR)
            print(f"\n  {GRN}✓ Traversal saved:{R}  {B}{saved_path}{R}")
            print(f"  {DIM}  {len(traversal['traversal'])} steps  │  "
                  f"{sum(1 for n in traversal['graph']['nodes'] if n['visited'])} nodes visited{R}\n")
        except Exception as e:
            print(f"\n  {YLW}Warning: traversal save failed: {e}{R}\n")
```

- [ ] **Step 2: Verify `chat.py` has no syntax errors**

```bash
cd /home/ankit/Unpod/super-sanyam/super/superdialog && python -c "import ast; ast.parse(open('chat.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run a short smoke test**

```bash
cd /home/ankit/Unpod/super-sanyam/super/superdialog
python chat.py
# Type 2-3 messages, then type: exit
```

Expected:
- After exiting: `✓ Traversal saved: .../Traversal_History/traversal_YYYYMMDD_HHMMSS.json`
- File exists: `ls src/superdialog/Traversal_History/`

- [ ] **Step 4: Verify JSON structure**

```bash
cd /home/ankit/Unpod/super-sanyam/super/superdialog
python -c "
import json, glob
files = sorted(glob.glob('src/superdialog/Traversal_History/traversal_*.json'))
data = json.load(open(files[-1]))
print('session_id:', data['session_id'])
print('steps:', len(data['traversal']))
print('graph nodes:', len(data['graph']['nodes']))
print('graph edges:', len(data['graph']['edges']))
print('traversal[0]:', json.dumps(data['traversal'][0], indent=2, ensure_ascii=False)[:300])
"
```

Expected: `session_id` present, `steps` >= 1, valid JSON structure, `traversal[0]` has `bot_message`, `criteria: null`.

---

## Task 5: Create `view_traversal.py`

**Files:**
- Create: `view_traversal.py` (next to `chat.py`)

- [ ] **Step 1: Create `view_traversal.py`**

```python
"""
view_traversal.py — render a superdialog traversal JSON as a PNG graph.

    python view_traversal.py                             # latest session
    python view_traversal.py traversal_20260522_143201.json  # specific file
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TRAVERSAL_DIR = Path(__file__).parent / "src/superdialog/traversal"


def load_traversal(arg: str | None) -> tuple[dict, Path]:
    """Return (traversal_dict, source_path)."""
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = TRAVERSAL_DIR / arg
        if not p.exists():
            print(f"File not found: {p}")
            sys.exit(1)
        return json.loads(p.read_text()), p

    files = sorted(TRAVERSAL_DIR.glob("traversal_*.json"))
    if not files:
        print(f"No traversal files in {TRAVERSAL_DIR}")
        print("Run: python chat.py  (then exit the session)")
        sys.exit(1)
    p = files[-1]
    return json.loads(p.read_text()), p


def render(data: dict, source_path: Path) -> Path:
    """Draw directed graph, save PNG next to source JSON."""
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Missing dependencies. Install with:")
        print("  pip install networkx matplotlib")
        sys.exit(1)

    G = nx.DiGraph()

    # Add nodes
    node_meta = {n["id"]: n for n in data["graph"]["nodes"]}
    for n in data["graph"]["nodes"]:
        G.add_node(n["id"], **n)

    # Add edges
    for e in data["graph"]["edges"]:
        G.add_edge(e["source"], e["target"], **e)

    # Layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=2.5)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_title(
        f"Flow traversal — {data['session_id']}  |  {data['flow_file']}  |  "
        f"complete={data['is_complete']}",
        fontsize=10,
    )

    # Node colours
    node_colors = []
    node_edge_colors = []
    for n_id in G.nodes:
        meta = node_meta.get(n_id, {})
        node_colors.append("#00cc44" if meta.get("visited") else "#cccccc")
        node_edge_colors.append("#cc0000" if meta.get("is_final") else "#333333")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        node_size=1800,
        linewidths=2.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight="bold")

    # Edges — split traversed vs not
    traversed_edges = [(e["source"], e["target"]) for e in data["graph"]["edges"] if e["traversed"]]
    untravelled_edges = [(e["source"], e["target"]) for e in data["graph"]["edges"] if not e["traversed"]]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=traversed_edges,
        edge_color="#0055cc",
        width=2.2,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=untravelled_edges,
        edge_color="#aaaaaa",
        width=1.0,
        style="dashed",
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.1",
    )

    # Edge step labels on traversed edges only
    edge_labels = {}
    edge_step_map = {(e["source"], e["target"]): e["traversed_at_step"] for e in data["graph"]["edges"] if e["traversed"]}
    for (src, tgt), step in edge_step_map.items():
        edge_labels[(src, tgt)] = f"step {step}"
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=6, label_pos=0.35)

    # Legend
    legend = [
        mpatches.Patch(color="#00cc44", label="Visited node"),
        mpatches.Patch(color="#cccccc", label="Unvisited node"),
        mpatches.Patch(color="#0055cc", label="Traversed edge"),
        mpatches.Patch(color="#aaaaaa", label="Untravelled edge"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8)
    ax.axis("off")

    out_path = source_path.parent / f"graph_{data['session_id']}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    data, source_path = load_traversal(arg)
    out = render(data, source_path)
    print(f"Graph saved: {out}")
    steps = len(data["traversal"])
    visited = sum(1 for n in data["graph"]["nodes"] if n["visited"])
    traversed = sum(1 for e in data["graph"]["edges"] if e["traversed"])
    total_edges = len(data["graph"]["edges"])
    print(f"  {steps} steps  |  {visited}/{len(data['graph']['nodes'])} nodes visited  |  {traversed}/{total_edges} edges traversed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd /home/ankit/Unpod/super-sanyam/super/superdialog && python -c "import ast; ast.parse(open('view_traversal.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Check networkx + matplotlib available**

```bash
python -c "import networkx, matplotlib; print('OK')"
```

If fails:
```bash
pip install networkx matplotlib
```

- [ ] **Step 4: Run view_traversal.py on saved traversal**

```bash
cd /home/ankit/Unpod/super-sanyam/super/superdialog && python view_traversal.py
```

Expected output:
```
Graph saved: .../Traversal_History/graph_YYYYMMDD_HHMMSS.png
  N steps  |  M/K nodes visited  |  P/Q edges traversed
```

- [ ] **Step 5: Open and verify PNG**

```bash
xdg-open src/superdialog/traversal/graph_*.png
```

Verify: green nodes for visited path, grey for unvisited, blue solid lines for traversed edges, grey dashed for untravelled.

---

## Self-Review Checklist

- [x] `build_traversal` sources: `TransitionRecord.criteria_met` (available), `visit_count` (available) — no unavailable fields
- [x] `save_traversal` creates dir if missing — matches spec error handling
- [x] `view_traversal.py` handles missing networkx with install hint — matches spec
- [x] `view_traversal.py` handles missing files — exits cleanly
- [x] `TRAVERSAL_DIR` constant used consistently in both `chat.py` and `view_traversal.py`
- [x] `chat_turns` step index aligns with `transition_log` — step 1 = initial entry (no transition), step 2+ = `transition_log[0]`, etc.
- [x] `criteria` is `None` for step 1 — matches schema
- [x] All spec fields present in output JSON