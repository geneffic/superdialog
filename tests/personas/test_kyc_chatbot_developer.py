"""Persona 1: KYC chatbot developer.

A fintech developer building a Know-Your-Customer bot. They:
1. Load a pre-authored flow from JSON.
2. Wire DialogMachine with a model URI.
3. Drive a multi-turn conversation through all nodes (greet → name → dob → pan → done).
4. Check that state advances correctly, slots accumulate, and the machine
   reaches the final node.
5. Use .assist() to inject mid-conversation context.
6. Use streaming mode for one turn.
7. Switch between flows in a FlowSet.
8. Reset and start a fresh conversation on the same instance.

All LLM calls are stubbed via a scripted provider so these tests are
fully hermetic — no API key needed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from superdialog import DialogMachine, Flow, FlowSet, StreamChunk, Turn
from superdialog.llm.provider import CompletionResult
from superdialog.llm.provider import StreamChunk as ProviderStreamChunk

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "flow"


class ScriptedProvider:
    """LLMProvider that pops responses from a script list."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def complete(
        self, messages: list[dict[str, Any]], tools: Any = None, **opts: Any
    ) -> CompletionResult:
        self.call_count += 1
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
    slots: dict[str, Any] | None = None,
    all_met: bool = True,
) -> str:
    return json.dumps(
        {
            "criteria_met": {},
            "extracted_slots": slots or {},
            "all_required_met": all_met,
            "user_insisting": False,
            "recommended_edge_id": edge,
            "reason": "stub",
            "response": response,
        }
    )


def _dm(responses: list[str], flow_path: str = "kyc.json") -> DialogMachine:
    provider = ScriptedProvider(responses)
    dm = DialogMachine(
        flow=Flow.load(FIXTURE_DIR / flow_path), llm="openai/gpt-4o-mini"
    )
    dm._llm = provider  # type: ignore[assignment]
    return dm


# ---------------------------------------------------------------------------
# Multi-turn: walk the KYC flow greet → name → dob → pan → done
# ---------------------------------------------------------------------------


async def test_full_kyc_journey_reaches_final_node() -> None:
    """Simulate a 4-turn KYC journey and verify the machine reaches 'done'."""
    dm = _dm(
        [
            # Turn 1: greet → collect_name (criteria eval + new-node reply)
            _criteria(edge="greet_to_name", response="Hi! What is your full name?"),
            "Please tell me your full name.",
            # Turn 2: collect_name → collect_dob
            _criteria(edge="name_to_dob", response="Got it", slots={"name": "Alice"}),
            "Thanks Alice. What is your date of birth?",
            # Turn 3: collect_dob → collect_pan
            _criteria(edge="dob_to_pan", response="Great", slots={"dob": "1990-05-15"}),
            "Now I need your PAN number.",
            # Turn 4: collect_pan → done
            _criteria(
                edge="pan_to_done", response="Thank you", slots={"pan": "ABCDE1234F"}
            ),
            "KYC complete. Thank you!",
        ]
    )

    await dm.turn("Hello")
    assert dm.state["node_id"] == "collect_name"

    await dm.turn("My name is Alice")
    assert dm.state["node_id"] == "collect_dob"
    assert "name" in dm.state["slots"]

    await dm.turn("15 May 1990")
    assert dm.state["node_id"] == "collect_pan"

    r4 = await dm.turn("ABCDE1234F")
    assert dm.state["node_id"] == "done"
    assert isinstance(r4, Turn)
    assert r4.metadata["outcome"] == "transition"


async def test_stay_when_criteria_not_met() -> None:
    """LLM says criteria not met → machine stays at current node."""
    dm = _dm(
        [_criteria(edge=None, response="Could you repeat your name?", all_met=False)]
    )
    result = await dm.turn("hmm")
    assert result.metadata["outcome"] == "stay"
    assert dm.state["node_id"] == "greet"
    assert "repeat" in result.text.lower()


# ---------------------------------------------------------------------------
# .assist() — mid-conversation context injection
# ---------------------------------------------------------------------------


async def test_assist_injects_system_message_before_next_turn() -> None:
    dm = _dm([_criteria(edge="greet_to_name", response="ok")])
    dm.assist("Customer is VIP. Be extra polite.")
    assert len(dm._pending_system_messages) == 1
    await dm.turn("hi")
    # After turn, pending messages are consumed
    assert len(dm._pending_system_messages) == 0


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_streaming_turn_yields_chunks_and_final_turn() -> None:
    dm = _dm(
        [
            _criteria(edge="greet_to_name", response="Hello Alice"),
            "Welcome to KYC.",
        ]
    )
    stream = await dm.turn("hi", stream=True)
    chunks: list[StreamChunk] = [c async for c in stream]
    assert chunks, "expected chunks"
    assert chunks[-1].done is True
    assert isinstance(chunks[-1].turn, Turn)
    reassembled = "".join(c.text for c in chunks)
    assert reassembled == chunks[-1].turn.text


# ---------------------------------------------------------------------------
# FlowSet: switch between flows
# ---------------------------------------------------------------------------


async def test_switch_flow_resets_to_new_initial_node() -> None:
    kyc = Flow.load(FIXTURE_DIR / "kyc.json")
    appt = Flow.load(FIXTURE_DIR / "appointment.json")
    dm = DialogMachine(
        flow=FlowSet({"kyc": kyc, "appointment": appt}),
        llm="openai/gpt-4o-mini",
    )
    dm._llm = ScriptedProvider(  # type: ignore[assignment]
        [_criteria(edge="greet_to_name", response="Hi")]
    )

    await dm.turn("hello")
    assert dm.state["node_id"] == "collect_name"

    dm.switch_flow("appointment")
    assert dm.state["node_id"] == "intro"


# ---------------------------------------------------------------------------
# Reset: start fresh on same instance
# ---------------------------------------------------------------------------


async def test_reset_returns_to_initial_node_with_clean_slots() -> None:
    dm = _dm(
        [
            _criteria(edge="greet_to_name", response="Hi", slots={"name": "Alice"}),
            "What is your name?",
        ]
    )
    await dm.turn("I'm Alice")
    assert dm.state["node_id"] == "collect_name"
    assert dm.state["slots"].get("name") == "Alice"

    dm.reset()
    assert dm.state["node_id"] == "greet"
    assert dm.state["slots"] == {}
