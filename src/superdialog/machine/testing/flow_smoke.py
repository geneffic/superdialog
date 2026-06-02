from __future__ import annotations

import asyncio
import sys
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from superdialog.machine.testing.mock_adapter import MockAdapter

if TYPE_CHECKING:
    from superdialog.flow.models import ConversationFlow


@dataclass(frozen=True)
class FlowSmokeResult:
    flow_path: str
    final_state: str
    is_complete: bool
    transitions: int
    edge_sequence: list[str]


def _ensure_livekit_sdk_imports_mocked() -> None:
    """Allow importing flow/machine modules without the livekit SDK installed."""
    for mod in [
        "livekit",
        "livekit.agents",
        "livekit.agents.llm",
        "livekit.agents.llm.tool_context",
        "livekit.agents.voice",
        "livekit.api",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


def _find_edge_path_to_final(flow: "ConversationFlow") -> list[str]:
    node_map = {n.id: n for n in flow.nodes}
    if flow.initial_node not in node_map:
        raise ValueError(f"Flow initial_node '{flow.initial_node}' not found in nodes")

    finals = {n.id for n in flow.nodes if n.is_final}
    if not finals:
        raise ValueError("Flow has no final nodes (is_final=true)")

    if flow.initial_node in finals:
        return []

    # BFS over node graph to find shortest path to any final node.
    queue: deque[str] = deque([flow.initial_node])
    visited: set[str] = {flow.initial_node}
    parent: dict[str, tuple[str, str]] = {}

    while queue:
        node_id = queue.popleft()
        node = node_map[node_id]
        for edge in node.edges:
            if not edge.target_node_id:
                continue
            if edge.target_node_id not in node_map:
                raise ValueError(
                    f"Edge '{edge.id}' from '{node_id}' points to missing "
                    f"node '{edge.target_node_id}'"
                )
            nxt = edge.target_node_id
            if nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = (node_id, edge.id)
            if nxt in finals:
                # reconstruct edge path
                edges: list[str] = []
                cur = nxt
                while cur != flow.initial_node:
                    prev, edge_id = parent[cur]
                    edges.append(edge_id)
                    cur = prev
                edges.reverse()
                return edges
            queue.append(nxt)

    raise ValueError(
        "No path from initial_node to any final node "
        f"(initial_node='{flow.initial_node}', finals={sorted(finals)})"
    )


async def smoke_test_flow_path_async(
    flow_path: str,
    *,
    max_steps: int = 50,
    user_message: str = "test",
) -> FlowSmokeResult:
    """Smoke test DialogStateMachine wiring for a flow JSON/YAML on disk.

    This test does NOT use an LLM. It finds a shortest graph path from the
    initial node to any final node, then drives the machine through that
    edge sequence using MockAdapter.
    """
    # Import locally so callers can use this helper even if livekit SDK isn't installed.
    _ensure_livekit_sdk_imports_mocked()
    from superdialog.flow.models import ConversationFlow
    from superdialog.machine.machine import DialogStateMachine

    flow = ConversationFlow.from_file(flow_path)
    edge_sequence = _find_edge_path_to_final(flow)
    if len(edge_sequence) > max_steps:
        raise ValueError(
            f"Computed edge path has {len(edge_sequence)} steps; max_steps={max_steps}"
        )

    adapter = MockAdapter(edge_sequence=edge_sequence)
    machine = await DialogStateMachine.from_flow(flow, adapter)

    for _ in edge_sequence:
        result = await machine.process_turn(user_message)
        if result.outcome != "transition":
            raise AssertionError(
                f"Expected transition, got {result.outcome} at state "
                f"'{result.from_node}' -> '{result.to_node}', edge={result.edge_id}"
            )

    if edge_sequence and not machine.is_complete:
        raise AssertionError(
            "Machine did not reach a final node after driving the edge sequence. "
            f"final_state='{machine.current_state}'"
        )

    if machine.is_complete and adapter.session_ended is not True:
        raise AssertionError("Expected adapter.end_session() to be called on final")

    return FlowSmokeResult(
        flow_path=flow_path,
        final_state=machine.current_state,
        is_complete=machine.is_complete,
        transitions=len(machine.context.transition_log),
        edge_sequence=edge_sequence,
    )


def smoke_test_flow_path(
    flow_path: str,
    *,
    max_steps: int = 50,
    user_message: str = "test",
) -> FlowSmokeResult:
    """Sync wrapper for `smoke_test_flow_path_async` (uses `asyncio.run`)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            smoke_test_flow_path_async(
                flow_path, max_steps=max_steps, user_message=user_message
            )
        )
    raise RuntimeError(
        "smoke_test_flow_path() cannot be called from an existing event loop; "
        "use smoke_test_flow_path_async() instead."
    )
