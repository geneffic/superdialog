"""
view_traversal.py — render a superdialog traversal JSON as a PNG graph.

    python view_traversal.py                                  # latest session
    python view_traversal.py traversal_20260522_143201.json   # specific file
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

    node_meta = {n["id"]: n for n in data["graph"]["nodes"]}
    for n in data["graph"]["nodes"]:
        G.add_node(n["id"], **n)

    for e in data["graph"]["edges"]:
        G.add_edge(e["source"], e["target"], **e)

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

    edge_labels = {}
    for e in data["graph"]["edges"]:
        if e["traversed"] and e["traversed_at_step"] is not None:
            edge_labels[(e["source"], e["target"])] = f"step {e['traversed_at_step']}"
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=6, label_pos=0.35)

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
