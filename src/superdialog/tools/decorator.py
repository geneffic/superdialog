"""@tool decorator — wraps a plain function as a PythonTool."""

from __future__ import annotations

from typing import Any, Callable, overload

from .python_tool import PythonTool


@overload
def tool(fn: Callable[..., Any]) -> PythonTool: ...


@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], PythonTool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> PythonTool | Callable[[Callable[..., Any]], PythonTool]:
    """Decorator that converts a function into a PythonTool.

    Usage::

        @tool
        async def lookup_customer(customer_id: str) -> dict:
            \"\"\"Look up a customer by ID.\"\"\"
            ...

        @tool(name="check_stock", description="Check inventory levels")
        async def check_inventory(sku: str) -> dict:
            ...

    The function name becomes the tool ``id`` unless overridden via ``name``.
    The schema is inferred from type hints; the description from the docstring.
    """

    def decorate(f: Callable[..., Any]) -> PythonTool:
        return PythonTool.of(f, name=name, description=description)

    if fn is not None:
        return decorate(fn)
    return decorate
