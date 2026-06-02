"""The :class:`Agent` Protocol — the conversation-engine contract.

A SessionWorker accepts any class that implements four members. The shipped
implementations are :class:`superdialog.DialogMachine` (flow-based,
opinionated), :class:`superdialog.agents.LLMAgent` (raw chat brain), and
:class:`superdialog.agents.LangChainAgent` (LangChain runnable wrapper).

This module deliberately ships no abstract base class — Python Protocols
keep third-party brains free to drop in without inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from .chat_context import ChatContext
from .stream import StreamChunk, ToolCall


@dataclass
class TurnResult:
    """Public turn output returned by :meth:`Agent.turn` (non-streaming)."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    """Conversation engine contract.

    Implementations hold per-conversation state in-process. ``SessionWorker``
    builds one Agent per Session via the configured ``agent_factory``.
    """

    async def turn(
        self,
        text: str,
        *,
        stream: bool = False,
    ) -> TurnResult | AsyncIterator[StreamChunk]: ...

    def assist(self, text: str) -> None:
        """Push a system-level instruction; takes effect next turn."""
        ...

    @property
    def chat_ctx(self) -> ChatContext: ...

    def load_chat_ctx(self, ctx: ChatContext) -> None: ...


__all__ = ["Agent", "TurnResult"]
