"""LangChainAgent — adapter over a LangChain runnable.

PORT NOTE: LangChain's runnable interface and message types shift between
releases (``langchain_core.runnables.Runnable``, message types like
``HumanMessage``, ``AIMessage``, ``SystemMessage``). This agent uses
duck-typing on the runnable's ``ainvoke`` method and converts to/from a
simple OpenAI-compatible message-dict shape so we don't pin a specific
LangChain version internally.

Required extra: ``pip install superdialog[langchain]``.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from ..agent import TurnResult
from ..chat_context import ChatContext, ChatMessage
from ..stream import StreamChunk


def _require_langchain() -> None:
    try:
        import langchain_core  # type: ignore  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "LangChainAgent requires the langchain extra: "
            "`pip install superdialog[langchain]`"
        ) from e


class LangChainAgent:
    """Wraps any LangChain runnable that accepts a list of messages.

    The runnable is expected to expose ``ainvoke(messages_list)`` and return
    either a string or a LangChain ``BaseMessage``-like object whose
    ``content`` is the assistant reply.
    """

    def __init__(
        self,
        runnable: Any,
        *,
        chat_ctx: ChatContext | None = None,
    ) -> None:
        _require_langchain()
        self._runnable = runnable
        self._chat: ChatContext = chat_ctx or ChatContext()

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat

    def load_chat_ctx(self, ctx: ChatContext) -> None:
        self._chat = ctx

    def assist(self, text: str) -> None:
        if not text:
            return
        self._chat.items.append(ChatMessage(role="system", content=text))

    async def turn(
        self,
        text: str,
        *,
        stream: bool = False,
    ) -> TurnResult | AsyncIterator[StreamChunk]:
        self._chat.items.append(ChatMessage(role="user", content=text))
        messages = [{"role": m.role, "content": m.content} for m in self._chat.items]
        invoke = getattr(self._runnable, "ainvoke", None)
        if invoke is None:
            raise RuntimeError(
                "LangChainAgent: runnable lacks `ainvoke`; pass an async runnable."
            )
        raw = await invoke(messages)
        reply = self._extract_reply(raw)
        self._chat.items.append(ChatMessage(role="assistant", content=reply))
        return TurnResult(text=reply, tool_calls=[], metadata={"backend": "langchain"})

    @staticmethod
    def _extract_reply(raw: Any) -> str:
        if isinstance(raw, str):
            return raw
        content = getattr(raw, "content", None)
        if isinstance(content, str):
            return content
        return str(raw)


__all__ = ["LangChainAgent"]
