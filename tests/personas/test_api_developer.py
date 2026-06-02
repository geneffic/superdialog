"""Persona 3: Quick-deploy API developer.

A developer who needs to expose a dialog over HTTP (FastAPI) and WebSocket
as fast as possible. They:
1. Mount a FastAPIRouter and hit /turn, /stream, /assist, /reset.
2. Wire a WebSocketRunner in single-tenant mode and drive it via JSON frames.
3. Wire a WebSocketRunner in multi-tenant (SessionWorker) mode and verify
   two clients are isolated.
4. Stream a turn over SSE and reassemble the chunks.

All LLM calls are stubbed via the shared FakeDialogMachine from conftest.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any, AsyncIterator

import pytest

from superdialog.stream import StreamChunk, Turn

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class FakeAgent:
    """Minimal Agent that scripts canned replies."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.received: list[str] = []
        self.assist_calls: list[str] = []
        self.reset_calls = 0

    def assist(self, text: str) -> None:
        self.assist_calls.append(text)

    def reset(self) -> None:
        self.reset_calls += 1

    async def turn(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> Any:
        self.received.append(text)
        turn = Turn(
            text=self.reply,
            tool_calls=[],
            metadata={"from_node": "n", "to_node": "n", "model": "fake"},
        )
        if not stream:
            return turn

        async def _gen() -> AsyncIterator[StreamChunk]:
            pieces = self.reply.split(" ")
            last = len(pieces) - 1
            for idx, piece in enumerate(pieces):
                yield StreamChunk(
                    text=piece if idx == last else f"{piece} ",
                    done=idx == last,
                    turn=turn if idx == last else None,
                )

        return _gen()

    @property
    def chat_ctx(self) -> Any:
        from superdialog.chat_context import ChatContext

        return ChatContext()

    def load_chat_ctx(self, ctx: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# FastAPI: /turn, /stream, /assist, /reset
# ---------------------------------------------------------------------------


fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient
fa_adapter = pytest.importorskip("superdialog.adapters.fastapi")


class TestFastAPIPersona:
    """A developer who mounts FastAPIRouter and exercises all endpoints."""

    def _app(self, agent: FakeAgent) -> Any:
        app = fastapi.FastAPI()
        fa_adapter.FastAPIRouter(agent).mount(app, prefix="/bot")
        return app

    def test_turn_returns_reply_and_metadata(self) -> None:
        agent = FakeAgent(reply="Your order shipped.")
        client = TestClient(self._app(agent))
        resp = client.post("/bot/turn", json={"text": "Where is my order?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["reply"] == "Your order shipped."
        assert "model" in body["metadata"]
        assert agent.received == ["Where is my order?"]

    def test_stream_sse_reassembles_to_full_reply(self) -> None:
        agent = FakeAgent(reply="Your order is on its way")
        client = TestClient(self._app(agent))
        with client.stream("POST", "/bot/stream", json={"text": "status?"}) as resp:
            raw = b"".join(resp.iter_bytes()).decode()
        # SSE lines are "data: {...}\n\n"
        events = [
            json.loads(line.removeprefix("data: "))
            for line in raw.strip().split("\n\n")
            if line.startswith("data: ")
        ]
        assert events, "expected SSE events"
        assert events[-1]["done"] is True
        reassembled = "".join(e["text"] for e in events)
        assert reassembled == "Your order is on its way"

    def test_assist_then_turn(self) -> None:
        agent = FakeAgent(reply="I understand you're frustrated.")
        client = TestClient(self._app(agent))
        resp = client.post("/bot/assist", json={"text": "Caller is angry"})
        assert resp.json() == {"status": "ok"}
        assert agent.assist_calls == ["Caller is angry"]

        resp = client.post("/bot/turn", json={"text": "This is unacceptable"})
        assert resp.status_code == 200

    def test_reset_clears_state(self) -> None:
        agent = FakeAgent()
        client = TestClient(self._app(agent))
        resp = client.post("/bot/reset")
        assert resp.json() == {"status": "ok"}
        assert agent.reset_calls == 1


# ---------------------------------------------------------------------------
# WebSocket: single-tenant
# ---------------------------------------------------------------------------

pytest.importorskip("websockets")
from superdialog.adapters.websocket import WebSocketRunner  # noqa: E402


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, data: str) -> None:
        self.sent.append(data)

    def messages(self) -> list[dict[str, Any]]:
        return [json.loads(s) for s in self.sent]


class TestWebSocketSingleTenant:
    """Developer serving a single DialogMachine over WS."""

    async def test_user_text_streams_chunks(self) -> None:
        agent = FakeAgent(reply="Hello there friend")
        runner = WebSocketRunner(agent=agent)
        ws = FakeWebSocket()
        await runner.handle_message(ws, json.dumps({"type": "user_text", "text": "hi"}))
        msgs = ws.messages()
        assert all(m["type"] == "agent_chunk" for m in msgs)
        final = next(m for m in msgs if m["done"])
        assert "metadata" in final
        reassembled = "".join(m["text"] for m in msgs)
        assert reassembled == "Hello there friend"

    async def test_assist_and_reset_lifecycle(self) -> None:
        agent = FakeAgent()
        runner = WebSocketRunner(agent=agent)
        ws = FakeWebSocket()

        await runner.handle_message(
            ws, json.dumps({"type": "assist", "text": "be empathetic"})
        )
        assert ws.messages() == [{"type": "assist_ack"}]
        assert agent.assist_calls == ["be empathetic"]

        ws2 = FakeWebSocket()
        await runner.handle_message(ws2, json.dumps({"type": "reset"}))
        assert ws2.messages() == [{"type": "reset_ack"}]
        assert agent.reset_calls == 1


# ---------------------------------------------------------------------------
# WebSocket: multi-tenant (SessionWorker)
# ---------------------------------------------------------------------------


class FakeSessionHandle:
    def __init__(self, agent: FakeAgent) -> None:
        self._agent = agent
        self.assist_calls: list[str] = []

    @property
    def agent(self) -> FakeAgent:
        return self._agent

    async def turn(self, text: str, *, stream: bool = False) -> Any:
        return await self._agent.turn(text, stream=stream)

    def assist(self, text: str) -> None:
        self.assist_calls.append(text)


class FakeSessionWorker:
    def __init__(self) -> None:
        self._sessions: dict[str, FakeSessionHandle] = {}

    @contextlib.asynccontextmanager
    async def acquire(self, session_id: str) -> Any:
        if session_id not in self._sessions:
            self._sessions[session_id] = FakeSessionHandle(
                FakeAgent(reply=f"reply for {session_id}")
            )
        yield self._sessions[session_id]


class TestWebSocketMultiTenant:
    """Developer serving multiple callers through SessionWorker."""

    async def test_two_clients_get_isolated_responses(self) -> None:
        worker = FakeSessionWorker()
        runner = WebSocketRunner(worker=worker)

        ws_a = FakeWebSocket()
        await runner.handle_message(
            ws_a,
            json.dumps({"type": "user_text", "session_id": "alice", "text": "hi"}),
        )
        ws_b = FakeWebSocket()
        await runner.handle_message(
            ws_b,
            json.dumps({"type": "user_text", "session_id": "bob", "text": "hey"}),
        )

        # Each session produced its own agent
        assert "alice" in worker._sessions
        assert "bob" in worker._sessions
        assert worker._sessions["alice"].agent.received == ["hi"]
        assert worker._sessions["bob"].agent.received == ["hey"]

        # Responses contain per-session text
        a_text = "".join(m["text"] for m in ws_a.messages())
        b_text = "".join(m["text"] for m in ws_b.messages())
        assert "alice" in a_text
        assert "bob" in b_text

    async def test_missing_session_id_rejected(self) -> None:
        runner = WebSocketRunner(worker=FakeSessionWorker())
        ws = FakeWebSocket()
        await runner.handle_message(ws, json.dumps({"type": "user_text", "text": "hi"}))
        assert ws.messages() == [{"type": "error", "message": "missing_session_id"}]

    async def test_assist_routed_to_session(self) -> None:
        worker = FakeSessionWorker()
        runner = WebSocketRunner(worker=worker)
        ws = FakeWebSocket()
        # create session first
        await runner.handle_message(
            ws,
            json.dumps({"type": "user_text", "session_id": "c1", "text": "hello"}),
        )
        ws2 = FakeWebSocket()
        await runner.handle_message(
            ws2,
            json.dumps({"type": "assist", "session_id": "c1", "text": "be patient"}),
        )
        assert ws2.messages() == [{"type": "assist_ack"}]
        assert worker._sessions["c1"].assist_calls == ["be patient"]
