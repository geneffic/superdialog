"""FastAPI adapter for any superdialog :class:`Agent`.

Provides :func:`make_router` (an ``APIRouter`` factory) and a
:class:`FastAPIRouter` convenience wrapper that knows how to mount the
router onto an existing :class:`fastapi.FastAPI` app.

Endpoints:

* ``POST /turn``   -- run one synchronous turn and return ``{reply, metadata}``.
* ``POST /stream`` -- run one streaming turn as Server-Sent Events.
* ``POST /assist`` -- queue a system-level steering instruction.
* ``POST /reset``  -- drop conversation memory (Agent.reset when available).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from superdialog.stream import StreamChunk

if TYPE_CHECKING:  # pragma: no cover
    from superdialog.agent import Agent
else:
    Agent = Any

logger = logging.getLogger(__name__)


def _require_fastapi() -> tuple[Any, Any, Any]:
    try:
        from fastapi import APIRouter, FastAPI  # type: ignore
        from fastapi.responses import StreamingResponse  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "FastAPI adapter requires the fastapi extra: "
            "`pip install superdialog[fastapi]`"
        ) from e
    return APIRouter, FastAPI, StreamingResponse


def make_router(agent: Agent) -> Any:
    """Return an ``APIRouter`` exposing the standard agent endpoints.

    Accepts any superdialog :class:`Agent`. ``/turn`` and ``/stream``
    are available unconditionally; ``/assist`` and ``/reset`` are wired
    in only when the underlying agent supports them (``DialogMachine``,
    ``LLMAgent`` do; bare protocol implementations may not).
    """
    APIRouter, _FastAPI, StreamingResponse = _require_fastapi()
    router = APIRouter()

    @router.post("/turn")
    async def handle_turn(payload: dict[str, Any]) -> dict[str, Any]:
        text = payload.get("text", "")
        context = payload.get("context") or None
        # Forward `context` only to agents that accept it (DialogMachine);
        # generic Agent.turn signature only takes (text, *, stream).
        try:
            turn = await agent.turn(text, context=context)  # type: ignore[call-arg]
        except TypeError:
            turn = await agent.turn(text)
        return {"reply": turn.text, "metadata": turn.metadata}

    @router.post("/stream")
    async def handle_stream(payload: dict[str, Any]) -> Any:
        text = payload.get("text", "")
        context = payload.get("context") or None
        try:
            stream = await agent.turn(text, context=context, stream=True)  # type: ignore[call-arg]
        except TypeError:
            stream = await agent.turn(text, stream=True)

        async def sse() -> Any:
            async for chunk in stream:  # type: ignore[union-attr]
                yield _sse_event(chunk)

        return StreamingResponse(sse(), media_type="text/event-stream")

    @router.post("/assist")
    async def handle_assist(payload: dict[str, Any]) -> dict[str, str]:
        text = payload.get("text", "")
        assist_fn = getattr(agent, "assist", None)
        if assist_fn is None:
            return {"status": "unsupported"}
        assist_fn(text)
        return {"status": "ok"}

    @router.post("/reset")
    async def handle_reset() -> dict[str, str]:
        reset_fn = getattr(agent, "reset", None)
        if reset_fn is None:
            return {"status": "unsupported"}
        reset_fn()
        return {"status": "ok"}

    return router


def _sse_event(chunk: StreamChunk) -> str:
    payload = {"text": chunk.text, "done": chunk.done}
    if chunk.done and chunk.turn is not None:
        payload["metadata"] = chunk.turn.metadata
    return f"data: {json.dumps(payload)}\n\n"


class FastAPIRouter:
    """Mountable router that exposes a superdialog :class:`Agent` over HTTP."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.router = make_router(agent)

    def mount(self, app: Any, prefix: str = "") -> None:
        """Attach the router to ``app`` (a ``FastAPI`` instance)."""
        app.include_router(self.router, prefix=prefix)


__all__ = ["FastAPIRouter", "make_router"]
