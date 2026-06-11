"""Mock adapters for testing the dialog state machine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from superdialog.machine.models import CriteriaResult

if TYPE_CHECKING:
    from superdialog.flow.models import CustomAction, FlowNode


class MockAdapter:
    """Mock adapter that cycles through a sequence of edge IDs.

    Each call to ``evaluate_criteria`` returns the next edge in the
    sequence with ``all_required_met=True``.  When the sequence is
    exhausted, ``all_required_met`` becomes ``False``.
    """

    def __init__(self, edge_sequence: list[str]) -> None:
        self._edge_sequence = list(edge_sequence)
        self._index = 0
        self.spoken: list[str] = []
        self.replies: list[str] = []
        self.actions_executed: list[str] = []
        self.session_ended: bool = False

    async def speak(self, text: str, node: FlowNode) -> None:
        """Record spoken text."""
        self.spoken.append(text)

    async def generate_reply(self, instruction: str, node: FlowNode, history: list[dict] | None = None, userdata: dict | None = None) -> str:
        """Return a fixed reply and record it."""
        reply = f"mock reply for {node.id}"
        self.replies.append(reply)
        return reply

    async def evaluate_criteria(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
        silent: bool = False,
    ) -> CriteriaResult:
        """Return the next edge in the sequence."""
        if self._index < len(self._edge_sequence):
            edge_id = self._edge_sequence[self._index]
            self._index += 1
            return CriteriaResult(
                node_id=node.id,
                criteria_met={"auto": True},
                all_required_met=True,
                recommended_edge_id=edge_id,
                reason="mock sequence",
            )
        return CriteriaResult(
            node_id=node.id,
            all_required_met=False,
            reason="sequence exhausted",
        )

    async def generate_recovery(self, node: FlowNode, error: str) -> str:
        """Return a mock recovery message."""
        return f"mock recovery for {node.id}"

    async def execute_action(
        self,
        action: CustomAction,
        userdata: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Record executed action and return None."""
        self.actions_executed.append(action.id)
        return None

    async def end_session(self) -> None:
        """Mark session as ended."""
        self.session_ended = True


class MockAdapterWithCriteria:
    """Mock adapter that always returns a fixed criteria result.

    Useful for testing specific transition scenarios.
    """

    def __init__(
        self,
        edge_id: str,
        criteria_met: dict[str, bool] | None = None,
        user_insisting: bool = False,
    ) -> None:
        self._edge_id = edge_id
        self._criteria_met = criteria_met or {"default": True}
        self._user_insisting = user_insisting
        self.spoken: list[str] = []
        self.session_ended: bool = False

    async def speak(self, text: str, node: FlowNode) -> None:
        """Record spoken text."""
        self.spoken.append(text)

    async def generate_reply(self, instruction: str, node: FlowNode, history: list[dict] | None = None, userdata: dict | None = None) -> str:
        """Return a fixed reply."""
        return f"fixed reply for {node.id}"

    async def evaluate_criteria(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
        silent: bool = False,
    ) -> CriteriaResult:
        """Return the fixed criteria result."""
        return CriteriaResult(
            node_id=node.id,
            criteria_met=self._criteria_met,
            all_required_met=all(self._criteria_met.values()),
            user_insisting=self._user_insisting,
            recommended_edge_id=self._edge_id,
            reason="fixed criteria",
        )

    async def generate_recovery(self, node: FlowNode, error: str) -> str:
        """Return a mock recovery message."""
        return f"fixed recovery for {node.id}"

    async def execute_action(
        self,
        action: CustomAction,
        userdata: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return None."""
        return None

    async def end_session(self) -> None:
        """Mark session as ended."""
        self.session_ended = True
