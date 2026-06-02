"""WebSocket adapter — drive the handler directly with a fake socket.

The wire protocol is exercised end-to-end without binding to a port.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

import pytest

pytest.importorskip("websockets")

from superdialog.adapters.websocket import WebSocketRunner  # noqa: E402


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, data: str) -> None:
        self.sent.append(data)

    def messages(self) -> list[dict[str, Any]]:
        return [json.loads(s) for s in self.sent]


@pytest.mark.asyncio
async def test_user_text_streams_back_chunks(fake_dm) -> None:
    runner = WebSocketRunner(fake_dm)
    ws = FakeWebSocket()
    await runner.handle_message(ws, json.dumps({"type": "user_text", "text": "hi"}))
    msgs = ws.messages()
    assert all(m["type"] == "agent_chunk" for m in msgs)
    assert any(m["done"] for m in msgs)
    final = next(m for m in msgs if m["done"])
    assert "metadata" in final
    assert fake_dm.received == ["hi"]


@pytest.mark.asyncio
async def test_reset_message_resets_machine(fake_dm) -> None:
    runner = WebSocketRunner(fake_dm)
    ws = FakeWebSocket()
    await runner.handle_message(ws, json.dumps({"type": "reset"}))
    assert ws.messages() == [{"type": "reset_ack"}]
    assert fake_dm.reset_calls == 1


@pytest.mark.asyncio
async def test_assist_message_forwards_to_agent(fake_dm) -> None:
    runner = WebSocketRunner(fake_dm)
    ws = FakeWebSocket()
    await runner.handle_message(ws, json.dumps({"type": "assist", "text": "be calm"}))
    assert ws.messages() == [{"type": "assist_ack"}]
    assert fake_dm.assist_calls == ["be calm"]


@pytest.mark.asyncio
async def test_invalid_json_returns_error(fake_dm) -> None:
    runner = WebSocketRunner(fake_dm)
    ws = FakeWebSocket()
    await runner.handle_message(ws, "not-json")
    assert ws.messages() == [{"type": "error", "message": "invalid_json"}]


@pytest.mark.asyncio
async def test_unknown_type_returns_error(fake_dm) -> None:
    runner = WebSocketRunner(fake_dm)
    ws = FakeWebSocket()
    await runner.handle_message(ws, json.dumps({"type": "ping"}))
    msgs = ws.messages()
    assert msgs == [{"type": "error", "message": "unknown_type:ping"}]


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_requires_exactly_one_of_agent_or_worker(fake_dm) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        WebSocketRunner()  # neither
    with pytest.raises(ValueError, match="exactly one"):
        WebSocketRunner(agent=fake_dm, worker=object())  # both


# ---------------------------------------------------------------------------
# Worker (multi-tenant) mode
# ---------------------------------------------------------------------------


class FakeSessionHandle:
    """Minimal handle exposing .turn, .assist, .agent for worker-mode tests."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self.assist_calls: list[str] = []

    @property
    def agent(self) -> Any:
        return self._agent

    async def turn(self, text: str, *, stream: bool = False) -> Any:
        return await self._agent.turn(text, stream=stream)

    def assist(self, text: str) -> None:
        self.assist_calls.append(text)


class FakeSessionWorker:
    """Minimal worker that hands out FakeSessionHandle per session_id."""

    def __init__(self, agent_factory: Any) -> None:
        self._factory = agent_factory
        self._sessions: dict[str, FakeSessionHandle] = {}

    @contextlib.asynccontextmanager
    async def acquire(self, session_id: str) -> Any:
        if session_id not in self._sessions:
            self._sessions[session_id] = FakeSessionHandle(self._factory())
        yield self._sessions[session_id]


@pytest.fixture()
def fake_worker() -> FakeSessionWorker:
    from tests.adapters.conftest import FakeDialogMachine

    return FakeSessionWorker(
        agent_factory=lambda: FakeDialogMachine(reply="session ok")
    )


@pytest.mark.asyncio
async def test_worker_mode_streams_with_session_id(fake_worker) -> None:
    runner = WebSocketRunner(worker=fake_worker)
    ws = FakeWebSocket()
    await runner.handle_message(
        ws, json.dumps({"type": "user_text", "session_id": "u1", "text": "hi"})
    )
    msgs = ws.messages()
    assert all(m["type"] == "agent_chunk" for m in msgs)
    assert any(m["done"] for m in msgs)


@pytest.mark.asyncio
async def test_worker_mode_isolates_sessions(fake_worker) -> None:
    runner = WebSocketRunner(worker=fake_worker)
    ws1, ws2 = FakeWebSocket(), FakeWebSocket()
    await runner.handle_message(
        ws1, json.dumps({"type": "user_text", "session_id": "u1", "text": "a"})
    )
    await runner.handle_message(
        ws2, json.dumps({"type": "user_text", "session_id": "u2", "text": "b"})
    )
    # Each session got its own agent
    assert "u1" in fake_worker._sessions
    assert "u2" in fake_worker._sessions
    h1 = fake_worker._sessions["u1"]
    h2 = fake_worker._sessions["u2"]
    assert h1.agent.received == ["a"]
    assert h2.agent.received == ["b"]


@pytest.mark.asyncio
async def test_worker_mode_rejects_missing_session_id(fake_worker) -> None:
    runner = WebSocketRunner(worker=fake_worker)
    ws = FakeWebSocket()
    await runner.handle_message(ws, json.dumps({"type": "user_text", "text": "hi"}))
    assert ws.messages() == [{"type": "error", "message": "missing_session_id"}]


@pytest.mark.asyncio
async def test_worker_mode_assist_routes_to_handle(fake_worker) -> None:
    runner = WebSocketRunner(worker=fake_worker)
    ws = FakeWebSocket()
    # First drive a turn to create the session
    await runner.handle_message(
        ws, json.dumps({"type": "user_text", "session_id": "u1", "text": "hi"})
    )
    ws2 = FakeWebSocket()
    await runner.handle_message(
        ws2, json.dumps({"type": "assist", "session_id": "u1", "text": "be nice"})
    )
    assert ws2.messages() == [{"type": "assist_ack"}]
    assert fake_worker._sessions["u1"].assist_calls == ["be nice"]
