"""RuntimeAdapter and NodeExecutor protocols.

RuntimeAdapter — legacy contract for criteria-based (text/batch) path.
NodeExecutor  — new contract for scope-based execution (LiveKit, Pipecat).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from superdialog.machine.models import CriteriaResult

if TYPE_CHECKING:
    from superdialog.flow.models import CustomAction, FlowNode
    from superdialog.machine.models import NodeScope


@runtime_checkable
class RuntimeAdapter(Protocol):
    """Contract that runtime adapters must implement.

    Used by the criteria-based (text/batch) execution path
    in DialogStateMachine.process_turn().
    """

    async def speak(self, text: str, node: FlowNode) -> None: ...

    async def generate_reply(self, instruction: str, node: FlowNode, history: list[dict[str, Any]] | None = None, userdata: dict[str, Any] | None = None) -> str: ...

    async def evaluate_criteria(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
        silent: bool = False,
    ) -> CriteriaResult: ...

    async def execute_action(
        self,
        action: CustomAction,
        userdata: dict[str, Any],
    ) -> dict[str, Any] | None: ...

    async def generate_recovery(self, node: FlowNode, error: str) -> str: ...

    async def end_session(self) -> None: ...


@runtime_checkable
class NodeExecutor(Protocol):
    """Contract for scope-based node executors.

    The machine builds a NodeScope and hands it to the executor.
    The executor operates within that scope:
      1. Speaks/displays node content
      2. Listens for user input
      3. Calls scope's request_transition() to move forward
      4. On allowed: receives new_scope, swaps to it
      5. On denied: continues with correction_hint guidance

    Executors are platform-specific (LiveKit, Pipecat, batch LLM)
    but all receive the same NodeScope contract from the machine.
    """

    async def execute_node(self, scope: NodeScope) -> None:
        """Start executing a node with the given scope.

        Called by the machine when entering a new node. The executor
        should deliver the node's content and begin listening.
        """
        ...

    async def swap_to_scope(self, scope: NodeScope) -> None:
        """Hot-swap to a new node scope after transition is approved.

        Called when request_transition() returns allowed=True. The
        executor should interrupt current activity and switch to the
        new scope's instructions, tools, and context.
        """
        ...

    async def shutdown(self) -> None:
        """Clean up when the conversation ends (final node reached)."""
        ...
