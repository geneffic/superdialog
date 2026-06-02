"""ToolRegistry -- central registry for machine tools."""

from __future__ import annotations

import logging
from typing import Any

from superdialog.machine.models import FlowContext, ToolDescriptor, ToolResult
from superdialog.machine.tools.base import MachineToolset

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for machine tools."""

    def __init__(self) -> None:
        self._tools: dict[str, MachineToolset] = {}

    def register(self, tool: MachineToolset) -> None:
        """Register a tool by its ID."""
        self._tools[tool.tool_id] = tool
        logger.debug(
            "[tools] registered tool=%s name=%s",
            tool.tool_id,
            tool.name,
        )

    def get(self, tool_id: str) -> MachineToolset | None:
        """Get a tool by ID."""
        return self._tools.get(tool_id)

    def has(self, tool_id: str) -> bool:
        """Check if a tool is registered."""
        return tool_id in self._tools

    def get_descriptors(self) -> list[ToolDescriptor]:
        """Get descriptors for all registered tools."""
        return [t.to_descriptor() for t in self._tools.values()]

    def get_node_tools(self, node_tool_ids: list[str]) -> list[ToolDescriptor]:
        """Get descriptors for tools assigned to a specific node."""
        return [
            self._tools[tid].to_descriptor()
            for tid in node_tool_ids
            if tid in self._tools
        ]

    async def execute(
        self,
        tool_id: str,
        args: dict[str, Any],
        context: FlowContext,
    ) -> ToolResult:
        """Execute a registered tool."""
        tool = self._tools.get(tool_id)
        if not tool:
            msg = f"Unknown tool: {tool_id}"
            raise ValueError(msg)
        return await tool.execute(args, context)
