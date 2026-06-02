"""HttpTool — POSTs args as JSON to a URL, returns parsed body."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .base import Tool, ToolResult


@dataclass
class HttpTool(Tool):
    id: str
    name: str
    description: str
    url: str
    method: str = "POST"
    auth: dict[str, Any] | None = None
    input_schema: dict[str, Any] | None = None

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        headers: dict[str, str] = {}
        if self.auth and self.auth.get("type") == "bearer":
            headers["Authorization"] = f"Bearer {self.auth['token']}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.request(self.method, self.url, json=args, headers=headers)
            if r.status_code >= 400:
                return ToolResult(data={}, error=f"HTTP {r.status_code}: {r.text}")
            try:
                payload = r.json()
            except Exception:
                payload = {"raw": r.text}
            if isinstance(payload, dict):
                return ToolResult(data=payload)
            return ToolResult(data={"value": payload})
