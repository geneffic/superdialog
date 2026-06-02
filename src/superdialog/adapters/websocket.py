"""WebSocket runner for any superdialog :class:`Agent` or :class:`SessionWorker`.

Two modes:

* **Single-tenant** (``WebSocketRunner(agent=...)``) -- one Agent multiplexed
  across all connections. State is shared; use only for demos / single-user
  servers.
* **Multi-tenant** (``WebSocketRunner(worker=...)``) -- session-keyed. Every
  inbound frame must carry a ``session_id``; the runner acquires it on the
  bound :class:`SessionWorker` and routes the message through the matching
  :class:`SessionHandle`. State is isolated per session_id and persisted via
  whatever ``SessionStore`` the worker is configured with.

Protocol (JSON messages over a single WS connection):

* Client → server::

    # single-tenant mode
    {"type": "user_text", "text": "..."}
    {"type": "assist",    "text": "..."}
    {"type": "reset"}

    # multi-tenant mode (session_id required on every frame)
    {"type": "user_text", "session_id": "user-42", "text": "..."}
    {"type": "assist",    "session_id": "user-42", "text": "..."}

* Server → client::

    {"type": "agent_chunk", "text": "...", "done": false}
    {"type": "agent_chunk", "text": "...", "done": true, "metadata": {...}}
    {"type": "assist_ack"}
    {"type": "reset_ack"}
    {"type": "error", "message": "..."}

Streaming is preferred; for single-shot replies callers can ignore all
but the final chunk (it carries the full ``metadata``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from superdialog.agent import Agent
    from superdialog.session import SessionWorker
else:
    Agent = Any
    SessionWorker = Any

logger = logging.getLogger(__name__)


def _require_websockets() -> Any:
    try:
        import websockets  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "WebSocketRunner requires the ws extra: `pip install superdialog[ws]`"
        ) from e
    return websockets


class WebSocketRunner:
    """Serve a superdialog :class:`Agent` or :class:`SessionWorker` over WS.

    Exactly one of ``agent`` / ``worker`` must be supplied. Mixing the two
    constructor forms raises ``ValueError`` at construction time so callers
    fail fast.
    """

    def __init__(
        self,
        agent: Agent | None = None,
        agent_id: str = "default",
        api_key: str | None = None,
        *,
        worker: SessionWorker | None = None,
    ) -> None:
        if (agent is None) == (worker is None):
            raise ValueError(
                "WebSocketRunner requires exactly one of `agent` or `worker`"
            )
        self.agent = agent
        self.worker = worker
        self.agent_id = agent_id
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def handle_message(self, ws: Any, raw: str) -> None:
        """Process one inbound JSON frame and stream the response."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send(json.dumps({"type": "error", "message": "invalid_json"}))
            return
        if self.worker is not None:
            await self._handle_with_worker(ws, msg)
        else:
            await self._handle_with_agent(ws, msg)

    async def _handle_with_agent(self, ws: Any, msg: dict[str, Any]) -> None:
        """Single-tenant dispatch -- everything routes through ``self.agent``."""
        kind = msg.get("type")
        if kind == "user_text":
            await self._stream_reply(ws, self.agent, msg.get("text", ""))
        elif kind == "assist":
            await self._call_assist(ws, self.agent, msg.get("text", ""))
        elif kind == "reset":
            await self._call_reset(ws, self.agent)
        else:
            await ws.send(
                json.dumps({"type": "error", "message": f"unknown_type:{kind}"})
            )

    async def _handle_with_worker(self, ws: Any, msg: dict[str, Any]) -> None:
        """Multi-tenant dispatch -- every frame must carry ``session_id``."""
        session_id = msg.get("session_id")
        if not session_id:
            await ws.send(
                json.dumps({"type": "error", "message": "missing_session_id"})
            )
            return
        kind = msg.get("type")
        async with self.worker.acquire(session_id) as handle:
            if kind == "user_text":
                await self._stream_reply(ws, handle, msg.get("text", ""))
            elif kind == "assist":
                handle.assist(msg.get("text", ""))
                await ws.send(json.dumps({"type": "assist_ack"}))
            elif kind == "reset":
                # SessionHandle has no reset(); fall back to the underlying agent.
                await self._call_reset(ws, handle.agent)
            else:
                await ws.send(
                    json.dumps({"type": "error", "message": f"unknown_type:{kind}"})
                )

    # ------------------------------------------------------------------
    # Per-target helpers (work against Agent OR SessionHandle)
    # ------------------------------------------------------------------

    async def _stream_reply(self, ws: Any, target: Any, text: str) -> None:
        stream = await target.turn(text, stream=True)
        async for chunk in stream:  # type: ignore[union-attr]
            frame: dict[str, Any] = {
                "type": "agent_chunk",
                "text": chunk.text,
                "done": chunk.done,
            }
            if chunk.done and chunk.turn is not None:
                frame["metadata"] = chunk.turn.metadata
            await ws.send(json.dumps(frame))

    async def _call_assist(self, ws: Any, target: Any, text: str) -> None:
        assist_fn = getattr(target, "assist", None)
        if assist_fn is None:
            await ws.send(
                json.dumps({"type": "error", "message": "assist_unsupported"})
            )
            return
        assist_fn(text)
        await ws.send(json.dumps({"type": "assist_ack"}))

    async def _call_reset(self, ws: Any, target: Any) -> None:
        reset_fn = getattr(target, "reset", None)
        if reset_fn is None:
            await ws.send(json.dumps({"type": "error", "message": "reset_unsupported"}))
            return
        reset_fn()
        await ws.send(json.dumps({"type": "reset_ack"}))

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _handler(self, ws: Any, *_: Any) -> None:
        try:
            async for raw in ws:
                await self.handle_message(ws, raw)
        except Exception as exc:  # noqa: BLE001 - protocol-level safety net
            logger.exception("websocket handler crashed: %s", exc)

    def serve(self, host: str = "0.0.0.0", port: int = 8080) -> None:  # nosec B104
        """Block on an asyncio event loop serving the WS endpoint."""
        websockets = _require_websockets()

        async def _main() -> None:
            async with websockets.serve(self._handler, host, port):
                await asyncio.Future()

        asyncio.run(_main())


__all__ = ["WebSocketRunner"]
