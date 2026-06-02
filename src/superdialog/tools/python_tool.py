"""PythonTool — wraps a callable (sync or async)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .base import Tool, ToolResult


@dataclass
class PythonTool(Tool):
    id: str
    name: str
    description: str
    fn: Callable[..., Any] | Callable[..., Awaitable[Any]] | None = None
    input_schema: dict[str, Any] | None = None

    @classmethod
    def of(
        cls,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> "PythonTool":
        """Convenience constructor: PythonTool.of(my_function)."""
        sig = inspect.signature(fn)
        return cls(
            id=name or fn.__name__,
            name=name or fn.__name__,
            description=description or (fn.__doc__ or "").strip(),
            fn=fn,
            input_schema=cls._infer_schema(sig),
        )

    @staticmethod
    def _infer_schema(sig: inspect.Signature) -> dict[str, Any]:
        type_map: dict[type, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }
        props: dict[str, dict[str, str]] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            ann = p.annotation
            jtype = type_map.get(ann, "string")
            props[pname] = {"type": jtype}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        return {"type": "object", "properties": props, "required": required}

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        if self.fn is None:
            return ToolResult(
                data={}, error=f"PythonTool {self.id!r} has no handler bound"
            )
        result = self.fn(**args)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, dict):
            return ToolResult(data=result)
        return ToolResult(data={"value": result})
