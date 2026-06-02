"""Group 8 — LangChainAgent contract test (skipped when extra missing)."""

from __future__ import annotations

import pytest

langchain_core = pytest.importorskip("langchain_core")

from superdialog.agents.langchain_agent import LangChainAgent  # noqa: E402
from superdialog.chat_context import ChatContext, ChatMessage  # noqa: E402


class _StubRunnable:
    """Minimal async runnable used to exercise the adapter without LangChain wiring."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_messages = None

    async def ainvoke(self, messages):
        self.last_messages = messages
        return self.reply


@pytest.mark.asyncio
async def test_langchain_agent_round_trip_with_string_reply() -> None:
    runnable = _StubRunnable("pong")
    agent = LangChainAgent(runnable)
    result = await agent.turn("ping")
    assert result.text == "pong"
    assert [m.role for m in agent.chat_ctx.items] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_langchain_agent_assist_appends_system_message() -> None:
    runnable = _StubRunnable("ok")
    agent = LangChainAgent(runnable)
    agent.assist("note")
    assert agent.chat_ctx.items == [ChatMessage("system", "note")]


def test_langchain_agent_load_chat_ctx_replaces() -> None:
    agent = LangChainAgent(_StubRunnable(""))
    new_ctx = ChatContext(items=[ChatMessage("user", "x")])
    agent.load_chat_ctx(new_ctx)
    assert agent.chat_ctx is new_ctx
