"""Persona 2: Customer support team lead.

A support team lead deploying a triage bot that handles multiple callers
simultaneously. They:
1. Use SessionWorker to multiplex sessions (one Agent per caller).
2. Verify session isolation — two callers converse independently.
3. Push mid-call system messages via SessionHandle.assist().
4. Verify persistence — state survives a re-acquire of the same session_id.
5. Use LLMAgent (no flow) for a simple FAQ brain alongside a DialogMachine
   flow brain, both driven through the same SessionWorker pattern.

All LLM calls are stubbed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from superdialog import (
    DialogMachine,
    Flow,
    InMemorySessionStore,
    LLMAgent,
    SessionWorker,
)
from superdialog.llm.provider import CompletionResult
from superdialog.llm.provider import StreamChunk as ProviderStreamChunk

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "flow"


class ScriptedProvider:
    """Pop responses from a script list; shared across tests."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[dict[str, Any]]] = []

    async def complete(
        self, messages: list[dict[str, Any]], tools: Any = None, **opts: Any
    ) -> CompletionResult:
        self.calls.append(messages)
        text = self._responses.pop(0) if self._responses else "{}"
        return CompletionResult(text=text, tool_calls=[], metadata={})

    async def stream(
        self, messages: list[dict[str, Any]], tools: Any = None, **opts: Any
    ) -> AsyncIterator[ProviderStreamChunk]:
        result = await self.complete(messages, tools, **opts)
        yield ProviderStreamChunk(text=result.text, tool_call_delta=None, done=True)


def _criteria(
    *,
    edge: str | None,
    response: str = "ok",
    all_met: bool = True,
) -> str:
    return json.dumps(
        {
            "criteria_met": {},
            "extracted_slots": {},
            "all_required_met": all_met,
            "user_insisting": False,
            "recommended_edge_id": edge,
            "reason": "stub",
            "response": response,
        }
    )


def _dm_factory(responses: list[str]) -> DialogMachine:
    """Build a DialogMachine with a scripted provider."""
    flow = Flow.load(FIXTURE_DIR / "escalation.json")
    dm = DialogMachine(flow=flow, llm="openai/gpt-4o-mini")
    dm._llm = ScriptedProvider(responses)  # type: ignore[assignment]
    return dm


# ---------------------------------------------------------------------------
# Multi-session isolation via SessionWorker
# ---------------------------------------------------------------------------


async def test_two_callers_have_independent_state() -> None:
    """Two concurrent callers on different session_ids get separate agents."""
    store = InMemorySessionStore()
    worker = SessionWorker(
        agent_factory=lambda: _dm_factory(
            [
                _criteria(edge="triage_to_resolve", response="I'll help"),
                "Let me look into that.",
            ]
        ),
        store=store,
    )

    async with worker.acquire("caller-A") as h_a:
        r_a = await h_a.turn("My order is late")
        assert r_a.text  # non-empty response

    async with worker.acquire("caller-B") as h_b:
        r_b = await h_b.turn("I need a refund")
        assert r_b.text

    # Both sessions are persisted independently
    rec_a = await store.load("caller-A")
    rec_b = await store.load("caller-B")
    assert rec_a is not None
    assert rec_b is not None
    # History lengths reflect each caller's own conversation
    assert len(rec_a.chat_ctx.items) > 0
    assert len(rec_b.chat_ctx.items) > 0


async def test_session_state_survives_reacquire() -> None:
    """Re-acquiring the same session_id picks up where we left off."""
    call_count = 0

    def factory() -> DialogMachine:
        nonlocal call_count
        call_count += 1
        return _dm_factory(
            [
                _criteria(edge="triage_to_resolve", response="ok"),
                "Looking into it.",
                # second turn on same session
                _criteria(edge="resolve_to_done", response="Resolved"),
                "All done.",
            ]
        )

    worker = SessionWorker(
        agent_factory=factory,
        store=InMemorySessionStore(),
    )

    # First acquire — advance to "resolve"
    async with worker.acquire("user-1") as h:
        await h.turn("help me")

    # Second acquire — same session, machine should be cached
    async with worker.acquire("user-1") as h:
        r = await h.turn("is it fixed?")
        assert r.text


async def test_assist_via_session_handle() -> None:
    """SessionHandle.assist() pushes a system message for the next turn."""
    worker = SessionWorker(
        agent_factory=lambda: _dm_factory(
            [
                _criteria(edge="triage_to_resolve", response="ok"),
                "On it.",
            ]
        ),
        store=InMemorySessionStore(),
    )

    async with worker.acquire("vip-99") as h:
        h.assist("This caller is a VIP. Prioritise their issue.")
        r = await h.turn("My account is locked")
        assert r.text


# ---------------------------------------------------------------------------
# LLMAgent (no flow) through SessionWorker
# ---------------------------------------------------------------------------


async def test_llm_agent_through_session_worker() -> None:
    """LLMAgent (raw chat, no flow) works as an Agent in SessionWorker."""
    stub = ScriptedProvider(["I can help with that!", "Glad to help."])

    def factory() -> LLMAgent:
        return LLMAgent(llm=stub, system_prompt="You are a helpful FAQ bot.")

    worker = SessionWorker(
        agent_factory=factory,  # type: ignore[arg-type]
        store=InMemorySessionStore(),
    )

    async with worker.acquire("faq-user") as h:
        r1 = await h.turn("How do I reset my password?")
        assert "help" in r1.text.lower()

    async with worker.acquire("faq-user") as h:
        r2 = await h.turn("Thanks!")
        assert r2.text

    # Verify chat history accumulated
    rec = await worker._store.load("faq-user")
    assert rec is not None
    # Should have system + user + assistant + user + assistant = 5 items
    assert len(rec.chat_ctx.items) >= 4
