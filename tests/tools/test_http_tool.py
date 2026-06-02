import httpx
import pytest

from superdialog.tools import HttpTool
from superdialog.tools import http_tool as http_tool_mod


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(http_tool_mod.httpx, "AsyncClient", _factory)


@pytest.mark.asyncio
async def test_http_tool_posts_and_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True, "id": 42})

    _patch_transport(monkeypatch, handler)

    tool = HttpTool(
        id="lookup", name="lookup", description="", url="https://example.com/api"
    )
    result = await tool.execute({"q": "x"})
    assert result.data == {"ok": True, "id": 42}
    assert seen["method"] == "POST"


@pytest.mark.asyncio
async def test_http_tool_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patch_transport(monkeypatch, handler)
    tool = HttpTool(id="x", name="x", description="", url="https://example.com/api")
    result = await tool.execute({})
    assert result.error is not None
    assert "500" in result.error


@pytest.mark.asyncio
async def test_http_tool_bearer_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json={"ok": True})

    _patch_transport(monkeypatch, handler)
    tool = HttpTool(
        id="x",
        name="x",
        description="",
        url="https://example.com/api",
        auth={"type": "bearer", "token": "secret"},
    )
    await tool.execute({})
    assert captured["auth"] == "Bearer secret"


@pytest.mark.asyncio
async def test_http_tool_non_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="plain text")

    _patch_transport(monkeypatch, handler)
    tool = HttpTool(id="x", name="x", description="", url="https://example.com/api")
    result = await tool.execute({})
    assert result.data == {"raw": "plain text"}
