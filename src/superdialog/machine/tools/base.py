"""MachineToolset -- provider-agnostic tool base for dialog_machine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from superdialog.machine.models import FlowContext, ToolDescriptor, ToolResult


class MachineToolset(ABC):
    """Provider-agnostic tool for dialog_machine.

    Subclass this to create tools that the machine can expose
    to any adapter (LiveKit, text, batch).
    """

    def __init__(self, tool_id: str, name: str, description: str) -> None:
        self.tool_id = tool_id
        self.name = name
        self.description = description

    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON schema for tool arguments."""
        ...

    @abstractmethod
    async def execute(
        self,
        args: dict[str, Any],
        context: FlowContext,
    ) -> ToolResult:
        """Execute tool, return result + optional transition edge."""
        ...

    def to_descriptor(self) -> ToolDescriptor:
        """Convert to provider-agnostic ToolDescriptor."""
        return ToolDescriptor(
            id=self.tool_id,
            description=self.description,
            is_data_collection=bool(self.input_schema().get("properties")),
            is_global=False,
            is_custom=True,
            input_schema=self.input_schema(),
            handler_id=self.tool_id,
        )
