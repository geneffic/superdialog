"""dialog_machine tools -- pluggable tool registry."""

from __future__ import annotations

from superdialog.machine.tools.base import MachineToolset
from superdialog.machine.tools.builtins import CalculatorTool, TimezoneTool
from superdialog.machine.tools.registry import ToolRegistry

__all__ = [
    "CalculatorTool",
    "MachineToolset",
    "TimezoneTool",
    "ToolRegistry",
    "build_default_registry",
]


def build_default_registry(
    *,
    include_builtins: bool = True,
) -> ToolRegistry:
    """Build a ToolRegistry with optional built-in tools."""
    registry = ToolRegistry()
    if include_builtins:
        registry.register(CalculatorTool())
        registry.register(TimezoneTool())
    return registry
