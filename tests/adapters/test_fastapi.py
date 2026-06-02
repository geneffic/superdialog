import importlib

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

fa_adapter = importlib.import_module("superdialog.adapters.fastapi")


def test_router_turn_endpoint(fake_dm) -> None:
    app = fastapi.FastAPI()
    fa_adapter.FastAPIRouter(fake_dm).mount(app, prefix="/dm")
    client = TestClient(app)

    resp = client.post("/dm/turn", json={"text": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "hello world"
    assert body["metadata"]["model"] == "fake"
    assert fake_dm.received == ["hi"]


def test_router_reset_endpoint(fake_dm) -> None:
    app = fastapi.FastAPI()
    fa_adapter.FastAPIRouter(fake_dm).mount(app, prefix="/dm")
    client = TestClient(app)

    resp = client.post("/dm/reset")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert fake_dm.reset_calls == 1


def test_router_assist_endpoint(fake_dm) -> None:
    app = fastapi.FastAPI()
    fa_adapter.FastAPIRouter(fake_dm).mount(app, prefix="/dm")
    client = TestClient(app)

    resp = client.post("/dm/assist", json={"text": "be calm"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert fake_dm.assist_calls == ["be calm"]


def test_router_assist_endpoint_unsupported_agent() -> None:
    """Agents without an ``assist`` method get a graceful 200 + status."""

    class BareAgent:
        async def turn(self, text, *, stream=False):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    app = fastapi.FastAPI()
    fa_adapter.FastAPIRouter(BareAgent()).mount(app, prefix="/dm")
    client = TestClient(app)

    resp = client.post("/dm/assist", json={"text": "x"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "unsupported"}


def test_router_stream_endpoint(fake_dm) -> None:
    app = fastapi.FastAPI()
    fa_adapter.FastAPIRouter(fake_dm).mount(app, prefix="/dm")
    client = TestClient(app)

    with client.stream("POST", "/dm/stream", json={"text": "hi"}) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode()
    assert "hello" in body
    assert "world" in body
    assert '"done": true' in body
