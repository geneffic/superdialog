"""DialogMachine-specific runtime state, externalised for persistence.

``FlowState`` is the wire-format companion to :class:`ChatContext` for sessions
backed by a :class:`DialogMachine`. It captures the flow-position, slot, and
audit fields needed to resume a conversation at the same point it was
suspended, without exposing the internal :class:`FlowContext` shape.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superdialog.machine.models import FlowContext, TransitionRecord


@dataclass
class FlowState:
    current_node_id: str = ""
    userdata: dict[str, Any] = field(default_factory=dict)
    node_slots: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_spoken_flags: dict[str, bool] = field(default_factory=dict)
    visit_count: dict[str, int] = field(default_factory=dict)
    transition_log: list["TransitionRecord"] = field(default_factory=list)

    @classmethod
    def from_flow_context(cls, ctx: "FlowContext") -> "FlowState":
        """Snapshot DM-specific state out of a live :class:`FlowContext`."""
        return cls(
            current_node_id=ctx.current_node_id,
            userdata=deepcopy(ctx.userdata),
            node_slots=deepcopy(ctx.node_slots),
            node_spoken_flags=dict(ctx.node_spoken_flags),
            visit_count=dict(ctx.visit_count),
            transition_log=list(ctx.transition_log),
        )

    def apply_to(self, ctx: "FlowContext") -> "FlowContext":
        """Mutate ``ctx`` in place with this state and return it."""
        ctx.current_node_id = self.current_node_id
        ctx.userdata = deepcopy(self.userdata)
        ctx.node_slots = deepcopy(self.node_slots)
        ctx.node_spoken_flags = dict(self.node_spoken_flags)
        ctx.visit_count = dict(self.visit_count)
        ctx.transition_log = list(self.transition_log)
        return ctx

    def to_flow_context_merged_with(self, ctx: "FlowContext") -> "FlowContext":
        """Apply this state on top of ``ctx`` and return the merged context."""
        return self.apply_to(ctx)


__all__ = ["FlowState"]
