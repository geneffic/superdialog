"""MachineHooks — optional preprocessing hooks for dialog machine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from superdialog.machine.models import FlowContext

logger = logging.getLogger(__name__)

# Type aliases for hook callables
InputHook = Callable[[str, FlowContext], str]
HistoryHook = Callable[[list[dict[str, Any]], FlowContext], list[dict[str, Any]]]
PromptHook = Callable[[str, FlowContext], str]
TransitionHook = Callable[[str, str, str, FlowContext], None]


@dataclass
class MachineHooks:
    """Optional preprocessing hooks for dialog machine.

    Each hook is a callable that transforms its input.
    All hooks are optional — None means passthrough.

    Hooks:
        preprocess_input: Transform user input before recording.
        preprocess_history: Transform history before LLM sees it.
        preprocess_prompt: Transform prompt after composition.
        on_transition: Called after a transition completes.
    """

    preprocess_input: InputHook | None = None
    preprocess_history: HistoryHook | None = None
    preprocess_prompt: PromptHook | None = None
    on_transition: TransitionHook | None = None

    def apply_input(self, user_input: str, context: FlowContext) -> str:
        """Apply input preprocessing hook (fail-open)."""
        if not self.preprocess_input:
            return user_input
        try:
            return self.preprocess_input(user_input, context)
        except Exception as exc:
            logger.warning("[hooks] preprocess_input failed: %s", exc)
            return user_input

    def apply_history(
        self,
        history: list[dict[str, Any]],
        context: FlowContext,
    ) -> list[dict[str, Any]]:
        """Apply history preprocessing hook (fail-open)."""
        if not self.preprocess_history:
            return history
        try:
            return self.preprocess_history(history, context)
        except Exception as exc:
            logger.warning("[hooks] preprocess_history failed: %s", exc)
            return history

    def apply_prompt(self, prompt: str, context: FlowContext) -> str:
        """Apply prompt preprocessing hook (fail-open)."""
        if not self.preprocess_prompt:
            return prompt
        try:
            return self.preprocess_prompt(prompt, context)
        except Exception as exc:
            logger.warning("[hooks] preprocess_prompt failed: %s", exc)
            return prompt

    def fire_transition(
        self,
        from_node: str,
        to_node: str,
        edge_id: str,
        context: FlowContext,
    ) -> None:
        """Fire transition hook (fail-open, non-blocking)."""
        if not self.on_transition:
            return
        try:
            self.on_transition(from_node, to_node, edge_id, context)
        except Exception as exc:
            logger.warning("[hooks] on_transition failed: %s", exc)
