"""Built-in tools for dialog_machine (ported from dograh)."""

from __future__ import annotations

import ast
import logging
import operator
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from superdialog.machine.models import FlowContext, ToolResult
from superdialog.machine.tools.base import MachineToolset

logger = logging.getLogger(__name__)

# Allowed AST node types for safe arithmetic
_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

# Safe constants
_SAFE_CONSTANTS = {"pi": 3.141592653589793, "e": 2.718281828459045}


def _safe_ast_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node with only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_ast_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        msg = f"Unsupported constant: {node.value}"
        raise ValueError(msg)
    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        msg = f"Unknown variable: {node.id}"
        raise ValueError(msg)
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            msg = f"Unsupported operator: {type(node.op).__name__}"
            raise ValueError(msg)
        left = _safe_ast_eval(node.left)
        right = _safe_ast_eval(node.right)
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            msg = f"Unsupported unary: {type(node.op).__name__}"
            raise ValueError(msg)
        return op_fn(_safe_ast_eval(node.operand))
    msg = f"Unsupported AST node: {type(node).__name__}"
    raise ValueError(msg)


class CalculatorTool(MachineToolset):
    """Safe arithmetic -- AST-based, no arbitrary code."""

    def __init__(self) -> None:
        super().__init__(
            "calculator",
            "safe_calculator",
            "Evaluate arithmetic expressions safely",
        )

    def input_schema(self) -> dict[str, Any]:
        """JSON schema for calculator tool arguments."""
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": ("Math expression (e.g., '2 + 3 * 4')"),
                }
            },
            "required": ["expression"],
        }

    async def execute(self, args: dict[str, Any], context: FlowContext) -> ToolResult:
        """Evaluate the arithmetic expression safely."""
        expr = args.get("expression", "")
        try:
            tree = ast.parse(expr, mode="eval")
            result = _safe_ast_eval(tree)
            return ToolResult(data={"result": result, "expression": expr})
        except Exception as exc:
            return ToolResult(data={"error": str(exc), "expression": expr})


class TimezoneTool(MachineToolset):
    """Get current time in a timezone."""

    def __init__(self) -> None:
        super().__init__(
            "timezone",
            "get_current_time",
            "Get current time in a timezone",
        )

    def input_schema(self) -> dict[str, Any]:
        """JSON schema for timezone tool arguments."""
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": ("IANA timezone (e.g., 'America/New_York')"),
                }
            },
            "required": ["timezone"],
        }

    async def execute(self, args: dict[str, Any], context: FlowContext) -> ToolResult:
        """Get the current time in the specified timezone."""
        tz_name = args.get("timezone", "UTC")
        try:
            tz = ZoneInfo(tz_name)
            now = datetime.now(tz)
            return ToolResult(
                data={
                    "time": now.strftime("%I:%M %p"),
                    "datetime": now.isoformat(),
                    "timezone": tz_name,
                }
            )
        except Exception as exc:
            return ToolResult(data={"error": str(exc), "timezone": tz_name})
