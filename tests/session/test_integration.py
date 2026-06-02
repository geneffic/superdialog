"""Group 11 — end-to-end exercises across the usage shapes."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from superdialog import (
    Agent,
    ChatContext,
    ChatMessage,
    DialogMachine,
    InMemorySessionStore,
    LLMAgent,
    NullSessionStore,
    SessionWorker,
)
from superdialog.flow.models import ConversationFlow, FlowNode
from superdialog.llm.provider import CompletionResult, StreamChunk


class _StubProvider:
    """Returns scripted replies and records calls."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.calls: list[list[dict[str, Any]]] = []

    async def complete(self, messages, tools=None, **opts) -> CompletionResult:
        self.calls.append(list(messages))
        return CompletionResult(
            text=self.reply, tool_calls=[], metadata={"model": "stub"}
        )

    async def stream(self, messages, tools=None, **opts) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text=self.reply, tool_call_delta=None, done=True)


def _kyc_flow() -> ConversationFlow:
    return ConversationFlow(
        id="kyc",
        initial_node="greet",
        system_prompt="kyc bot",
        nodes=[
            FlowNode(id="greet", name="Greet", static_text="Hello!", is_final=True),
        ],
    )


# ---- Shape D: SessionWorker with LLMAgent (no DM) ----------------------------


@pytest.mark.asyncio
async def test_shape_d_session_worker_with_llm_agent_persists_history() -> None:
    """LLMAgent + InMemorySessionStore: resume across acquires."""
    provider = _StubProvider(reply="hello-world")
    store = InMemorySessionStore()
    worker = SessionWorker(
        agent_factory=lambda: LLMAgent(llm=provider),
        store=store,
    )

    async with worker.acquire("u-1") as h:
        await h.turn("hi")

    await worker.close_session("u-1")

    async with worker.acquire("u-1") as h:
        roles = [m.role for m in h.agent.chat_ctx.items]
    # Resumed session should have the previous user+assistant pair.
    assert "user" in roles
    assert "assistant" in roles


# ---- Shape C: SessionWorker with NullSessionStore (voice-style, no persist) --


@pytest.mark.asyncio
async def test_shape_c_null_store_drops_state_between_acquires() -> None:
    """NullSessionStore: each acquire of the same id is a fresh session."""
    provider = _StubProvider(reply="x")
    worker = SessionWorker(
        agent_factory=lambda: LLMAgent(llm=provider),
        store=NullSessionStore(),
    )

    async with worker.acquire("call-1") as h:
        await h.turn("first")

    await worker.close_session("call-1")

    async with worker.acquire("call-1") as h:
        roles = [m.role for m in h.agent.chat_ctx.items]
    # Fresh session — no prior history.
    assert roles == []


# ---- Shape A/B: SessionWorker with DialogMachine facade ----------------------


@pytest.mark.asyncio
async def test_shape_a_dialog_machine_facade_satisfies_agent_protocol() -> None:
    """DialogMachine (the public facade) is usable as an Agent."""
    dm = DialogMachine(flow=_kyc_flow(), llm="openai/gpt-4o-mini")
    assert isinstance(dm, Agent)
    # chat_ctx + assist work pre-bootstrap (lazy).
    dm.assist("Be brief.")
    assert any(
        m.role == "system" and m.content == "Be brief." for m in dm.chat_ctx.items
    )


@pytest.mark.asyncio
async def test_shape_a_session_worker_with_dm_facade_factory() -> None:
    """SessionWorker can be wired with a DialogMachine factory; chat_ctx flows."""
    worker = SessionWorker(
        agent_factory=lambda: DialogMachine(flow=_kyc_flow(), llm="openai/gpt-4o-mini"),
        store=InMemorySessionStore(),
    )
    # Just exercise the lifecycle; do not actually turn (would call LLM).
    async with worker.acquire("conv-1") as h:
        h.assist("test")
    record = await worker._store.load("conv-1")  # type: ignore[attr-defined]
    assert record is not None
    # The system message we injected should be persisted.
    assert any(m.role == "system" for m in record.chat_ctx.items)


# ---- Shape E: Direct DialogMachine (today's quickstart, unchanged) ----------


@pytest.mark.asyncio
async def test_shape_e_direct_dialog_machine_quickstart_unchanged() -> None:
    """The headline quickstart still works after the additive changes."""
    dm = DialogMachine(flow=_kyc_flow(), llm="openai/gpt-4o-mini")
    # Direct chat_ctx access (new in v0.2).
    ctx = dm.chat_ctx
    assert isinstance(ctx, ChatContext)
    dm.assist("Be brief.")
    assert dm.chat_ctx.items[-1] == ChatMessage("system", "Be brief.")
    # turn() is unchanged in signature; we don't call it here to avoid LLM I/O.


def test_llm_agent_satisfies_agent_protocol() -> None:
    agent = LLMAgent(llm=_StubProvider())
    assert isinstance(agent, Agent)
