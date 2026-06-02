"""MCPTool — proxies tool calls to an MCP server.

PORT NOTE: the live ``mcp`` client API is in flux. The implementation here
opens an SSE session lazily and forwards ``execute(args)`` as
``session.call_tool(self.id, args)``. Verify against the real ``mcp`` package
before v0.1 ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import Tool, ToolResult


@dataclass
class MCPTool(Tool):
    id: str
    name: str
    description: str
    server: str
    input_schema: dict[str, Any] | None = None
    _session: Any = field(default=None, init=False, repr=False)
    _ctx: Any = field(default=None, init=False, repr=False)

    async def _ensure_connected(self) -> None:
        if self._session is not None:
            return
        try:
            from mcp.client.session import ClientSession  # type: ignore
            from mcp.client.sse import sse_client  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "MCPTool requires the `mcp` extra: pip install superdialog[mcp]"
            ) from e
        self._ctx = sse_client(self.server)
        read, write = await self._ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        await self._ensure_connected()
        result = await self._session.call_tool(self.id, args)
        contents = getattr(result, "content", None) or []
        if not contents:
            return ToolResult(data={})
        first = contents[0]
        text = getattr(first, "text", None)
        if text is None:
            return ToolResult(data={"value": str(first)})
        return ToolResult(data={"value": text})
