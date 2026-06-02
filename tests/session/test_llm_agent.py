"""Group 7 — LLMAgent contract tests with a stub provider."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from superdialog.agent import Agent, TurnResult
from superdialog.agents.llm_agent import LLMAgent
from superdialog.chat_context import ChatContext, ChatMessage
from superdialog.llm.provider import CompletionResult, StreamChunk


class _StubProvider:
    """Records messages received and emits a scripted reply."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.last_messages: list[dict[str, Any]] = []

    async def complete(self, messages, tools=None, **opts) -> CompletionResult:
        self.last_messages = messages
        return CompletionResult(
            text=self.reply, tool_calls=[], metadata={"model": "stub"}
        )

    async def stream(self, messages, tools=None, **opts) -> AsyncIterator[StreamChunk]:
        for ch in self.reply.split():
            yield StreamChunk(text=ch + " ", tool_call_delta=None, done=False)
        yield StreamChunk(text=None, tool_call_delta=None, done=True)


def test_llm_agent_satisfies_agent_protocol() -> None:
    agent = LLMAgent(llm=_StubProvider())
    assert isinstance(agent, Agent)


@pytest.mark.asyncio
async def test_llm_agent_records_user_and_assistant_in_chat_ctx() -> None:
    provider = _StubProvider(reply="hi there")
    agent = LLMAgent(llm=provider)
    result = await agent.turn("hello")
    assert isinstance(result, TurnResult)
    assert result.text == "hi there"
    roles = [m.role for m in agent.chat_ctx.items]
    assert roles == ["user", "assistant"]
    assert agent.chat_ctx.items[1].content == "hi there"


@pytest.mark.asyncio
async def test_llm_agent_includes_system_prompt() -> None:
    provider = _StubProvider()
    agent = LLMAgent(llm=provider, system_prompt="be helpful")
    await agent.turn("hi")
    assert provider.last_messages[0] == {"role": "system", "content": "be helpful"}


@pytest.mark.asyncio
async def test_llm_agent_assist_appends_system_message() -> None:
    agent = LLMAgent(llm=_StubProvider())
    agent.assist("note")
    assert agent.chat_ctx.items[-1] == ChatMessage("system", "note")


@pytest.mark.asyncio
async def test_llm_agent_load_chat_ctx_replaces_history() -> None:
    agent = LLMAgent(llm=_StubProvider())
    new_ctx = ChatContext(items=[ChatMessage("user", "x")])
    agent.load_chat_ctx(new_ctx)
    assert agent.chat_ctx is new_ctx
