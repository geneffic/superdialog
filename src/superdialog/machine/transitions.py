"""TransitionEngine — pytransitions wrapper for dialog state machine."""

from __future__ import annotations

import logging
from typing import Any

from transitions.extensions.asyncio import AsyncMachine

from superdialog.flow.models import ConversationFlow

logger = logging.getLogger(__name__)


class TransitionEngine:
    """Wraps pytransitions AsyncMachine for FSM state transitions.

    Extracted from DialogStateMachine.from_flow() to isolate
    state management from business logic.
    """

    # pytransitions uses `state` attribute on the model
    state: str = ""

    def __init__(self, flow: ConversationFlow) -> None:
        states: list[dict[str, Any]] = [{"name": node.id} for node in flow.nodes]

        transitions: list[dict[str, Any]] = []
        for node in flow.nodes:
            for edge in node.edges:
                if edge.target_node_id:
                    transitions.append(
                        {
                            "trigger": edge.id,
                            "source": node.id,
                            "dest": edge.target_node_id,
                        }
                    )

        # Register global edges from every interruptible non-final node
        for gedge in flow.global_edges:
            if not gedge.target_node_id:
                continue
            for node in flow.nodes:
                if node.is_final or not node.interruptible:
                    continue
                local_ids = {e.id for e in node.edges}
                if gedge.id in local_ids:
                    continue
                transitions.append(
                    {
                        "trigger": gedge.id,
                        "source": node.id,
                        "dest": gedge.target_node_id,
                    }
                )

        self._machine = AsyncMachine(
            model=self,
            states=states,
            transitions=transitions,
            initial=flow.initial_node,
            auto_transitions=False,
            queued=True,
        )

        logger.info(
            "[transitions] built engine: states=%d transitions=%d initial=%s",
            len(states),
            len(transitions),
            flow.initial_node,
        )

    @property
    def current_state(self) -> str:
        """Current state ID."""
        return self.state

    def get_available_triggers(self) -> list[str]:
        """Edge IDs valid from current state."""
        return self._machine.get_triggers(self.state)

    def is_valid_trigger(self, edge_id: str) -> bool:
        """Check if edge_id is a valid trigger from current state."""
        return edge_id in self.get_available_triggers()

    async def fire(self, edge_id: str) -> None:
        """Execute a transition trigger."""
        trigger_fn = getattr(self, edge_id)
        await trigger_fn()

    def set_state(self, state_id: str) -> None:
        """Directly set state (for restoring from persistence)."""
        self.state = state_id
