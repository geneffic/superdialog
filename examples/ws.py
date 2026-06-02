"""WebSocket examples: single-tenant and session-aware modes.

Run with ``pip install superdialog[ws]`` and ``python examples/ws.py``.

Single-tenant (default):
    One ``DialogMachine`` shared across all WS connections. Good for
    demos and single-user CLIs.

Session-aware (``--sessions``):
    Each client frame carries a ``session_id``; state is isolated per
    session via ``SessionWorker`` + ``InMemorySessionStore``.
"""

from __future__ import annotations

import sys

from superdialog import DialogMachine, InMemorySessionStore, SessionWorker
from superdialog.adapters.websocket import WebSocketRunner
from superdialog.flow.models import ConversationFlow, FlowNode


def _build_flow() -> ConversationFlow:
    return ConversationFlow(
        id="demo",
        initial_node="welcome",
        nodes=[FlowNode(id="welcome", static_text="Hi.", is_final=True)],
    )


def main() -> None:
    sessions = "--sessions" in sys.argv

    if sessions:
        # Multi-tenant: one Agent per session, state isolated + persisted.
        flow = _build_flow()
        worker = SessionWorker(
            agent_factory=lambda: DialogMachine(flow=flow, llm="openai/gpt-5.1"),
            store=InMemorySessionStore(),
        )
        print("Serving WS (session-aware) on 0.0.0.0:8080")  # noqa: T201
        WebSocketRunner(worker=worker).serve(host="0.0.0.0", port=8080)  # nosec B104
    else:
        # Single-tenant: one shared DialogMachine.
        dm = DialogMachine(flow=_build_flow(), llm="openai/gpt-5.1")
        print("Serving WS (single-tenant) on 0.0.0.0:8080")  # noqa: T201
        WebSocketRunner(agent=dm).serve(host="0.0.0.0", port=8080)  # nosec B104


if __name__ == "__main__":
    main()
