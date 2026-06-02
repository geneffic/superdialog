"""Tool ABC + JSON deserializer.

Subclasses (PythonTool, HttpTool, MCPTool) live in sibling modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    data: dict[str, Any]
    transition_edge_id: str | None = None
    error: str | None = None


class Tool(ABC):
    id: str
    name: str
    description: str
    input_schema: dict[str, Any] | None

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> ToolResult: ...

    def to_openai_function(self) -> dict[str, Any]:
        """Render this tool as an OpenAI function-tool spec."""
        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }

    @staticmethod
    def from_dict(
        spec: dict[str, Any],
        handler_registry: dict[str, Callable[..., Any]] | None = None,
    ) -> "Tool":
        """Deserialize JSON tool entry into the right subclass via `type`."""
        from .http_tool import HttpTool
        from .mcp_tool import MCPTool
        from .python_tool import PythonTool

        ttype = spec.get("type", "python")
        if ttype == "python":
            handler = (handler_registry or {}).get(spec.get("handler_id", ""))
            return PythonTool(
                id=spec["id"],
                name=spec.get("name", spec["id"]),
                description=spec.get("description", ""),
                input_schema=spec.get("input_schema"),
                fn=handler,
            )
        if ttype == "http":
            return HttpTool(
                id=spec["id"],
                name=spec.get("name", spec["id"]),
                description=spec.get("description", ""),
                input_schema=spec.get("input_schema"),
                url=spec["url"],
                method=spec.get("method", "POST"),
                auth=spec.get("auth"),
            )
        if ttype == "mcp":
            return MCPTool(
                id=spec["id"],
                name=spec.get("name", spec["id"]),
                description=spec.get("description", ""),
                server=spec["server"],
                input_schema=spec.get("input_schema"),
            )
        raise ValueError(f"Unknown tool type: {ttype}")
