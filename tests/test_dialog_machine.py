"""Tests for the ``DialogMachine`` facade.

We exercise the spec-aligned API surface (``turn``, ``inject_system``,
``reset``, ``set_llm``, ``switch_flow``, ``state``) with a stub
``LLMProvider`` so the tests run hermetically without network calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from superdialog import DialogMachine, Flow, FlowSet, Turn
from superdialog.llm.provider import CompletionResult
from superdialog.llm.provider import StreamChunk as _PSChunk

FIXTURE = Path(__file__).parent / "fixtures" / "flow" / "kyc.json"


class StubProvider:
    """Scripted ``LLMProvider`` for hermetic tests.

    Returns successive responses from ``responses`` for each ``complete``
    call. CriteriaJudge expects valid JSON, so each entry is a JSON
    string that drives the desired transition.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        self.calls.append(messages)
        if self._responses:
            text = self._responses.pop(0)
        else:
            text = "{}"
        return CompletionResult(text=text, tool_calls=[], metadata={})

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[_PSChunk]:
        result = await self.complete(messages, tools, **opts)
        yield _PSChunk(text=result.text, tool_call_delta=None, done=True)


def _criteria_json(
    *,
    edge_id: str | None,
    response: str = "ok",
    extracted: dict[str, Any] | None = None,
    all_met: bool = True,
) -> str:
    import json

    return json.dumps(
        {
            "criteria_met": {},
            "extracted_slots": extracted or {},
            "all_required_met": all_met,
            "user_insisting": False,
            "recommended_edge_id": edge_id,
            "reason": "stub",
            "response": response,
        }
    )


def _load_flow() -> Flow:
    return Flow.load(FIXTURE)


def _machine_with(responses: list[str]) -> tuple[DialogMachine, StubProvider]:
    """Return a DialogMachine wired to a StubProvider.

    We bypass ``resolve_llm`` by overwriting ``_llm`` after construction.
    """
    provider = StubProvider(responses)
    machine = DialogMachine(flow=_load_flow(), llm="openai/gpt-4o-mini")
    machine._llm = provider  # type: ignore[assignment]
    return machine, provider


# ---------------------------------------------------------------------------
# turn() -- non-streaming
# ---------------------------------------------------------------------------


async def test_turn_returns_turn_and_advances_state() -> None:
    machine, _ = _machine_with(
        [_criteria_json(edge_id="greet_to_name", response="Hi Alice")]
    )
    result = await machine.turn("Hello, my name is Alice")
    assert isinstance(result, Turn)
    assert result.text  # non-empty (response or generated_reply)
    assert result.metadata["from_node"] == "greet"
    assert result.metadata["to_node"] == "collect_name"
    assert result.metadata["outcome"] == "transition"
    assert machine.state["node_id"] == "collect_name"


async def test_turn_stays_when_no_edge_recommended() -> None:
    machine, _ = _machine_with(
        [_criteria_json(edge_id=None, response="Could you repeat that?", all_met=False)]
    )
    result = await machine.turn("uh")
    assert result.metadata["outcome"] == "stay"
    assert result.metadata["from_node"] == result.metadata["to_node"] == "greet"
    assert "repeat" in result.text.lower()


# ---------------------------------------------------------------------------
# inject_system / reset
# ---------------------------------------------------------------------------


async def test_assist_flushes_into_history_on_next_turn() -> None:
    machine, provider = _machine_with(
        [_criteria_json(edge_id="greet_to_name", response="Hi")]
    )
    machine.assist("Caller is upset. Be empathetic.")
    await machine.turn("hi")
    history = provider.calls[0]
    assert any(
        m.get("role") == "system" and "empathetic" in m.get("content", "")
        for m in history
    )


async def test_reset_clears_machine_and_memory() -> None:
    machine, _ = _machine_with([_criteria_json(edge_id="greet_to_name", response="Hi")])
    await machine.turn("hello")
    assert machine.state["node_id"] == "collect_name"
    machine.reset()
    # next turn rebuilds at the flow's initial node
    assert machine.state["node_id"] == "greet"


# ---------------------------------------------------------------------------
# set_llm / switch_flow
# ---------------------------------------------------------------------------


async def test_set_llm_swaps_provider_on_active_adapter() -> None:
    machine, first = _machine_with(
        [_criteria_json(edge_id="greet_to_name", response="Hi")]
    )
    await machine.turn("hello")  # build the adapter
    second = StubProvider(
        [
            _criteria_json(
                edge_id="name_to_dob", response="ok", extracted={"name": "Alice"}
            )
        ]
    )
    machine.set_llm("openai/gpt-4o-mini")
    # set_llm just rebuilt _llm via resolve; overwrite for hermetic test
    machine._llm = second  # type: ignore[assignment]
    assert machine._adapter is not None
    machine._adapter.set_provider(second)
    result = await machine.turn("Alice")
    assert result.metadata["to_node"] == "collect_dob"
    assert second.calls, "new provider should have been called"


async def test_switch_flow_routes_to_named_flow() -> None:
    flow_a = _load_flow()
    flow_b = _load_flow()
    machine = DialogMachine(
        flow=FlowSet({"main": flow_a, "alt": flow_b}),
        llm="openai/gpt-4o-mini",
    )
    machine._llm = StubProvider(  # type: ignore[assignment]
        [_criteria_json(edge_id="greet_to_name", response="Hi")]
    )
    await machine.turn("hello")
    assert machine.state["node_id"] == "collect_name"
    machine.switch_flow("alt")
    # alt rebuilds clean
    assert machine.state["node_id"] == "greet"
    with pytest.raises(KeyError):
        machine.switch_flow("does_not_exist")


# ---------------------------------------------------------------------------
# state property
# ---------------------------------------------------------------------------


def test_state_before_first_turn_returns_initial_node() -> None:
    machine = DialogMachine(flow=_load_flow(), llm="openai/gpt-4o-mini")
    snapshot = machine.state
    assert snapshot == {"node_id": "greet", "slots": {}}
