"""Tests for ``DialogMachine.turn(stream=...)``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from superdialog import DialogMachine, Flow, StreamChunk, Turn
from superdialog.llm.provider import CompletionResult

FIXTURE = Path(__file__).parent / "fixtures" / "flow" / "kyc.json"


class StubProvider:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        text = self._responses.pop(0) if self._responses else "{}"
        return CompletionResult(text=text, tool_calls=[], metadata={})

    async def stream(self, *args: Any, **kwargs: Any):  # noqa: D401 -- unused
        raise NotImplementedError


def _criteria_json(*, edge_id: str | None, response: str) -> str:
    return json.dumps(
        {
            "criteria_met": {},
            "extracted_slots": {},
            "all_required_met": True,
            "user_insisting": False,
            "recommended_edge_id": edge_id,
            "reason": "stub",
            "response": response,
        }
    )


async def test_stream_text_yields_chunks_with_final_turn() -> None:
    machine = DialogMachine(flow=Flow.load(FIXTURE), llm="openai/gpt-4o-mini")
    machine._llm = StubProvider(  # type: ignore[assignment]
        [_criteria_json(edge_id="greet_to_name", response="Welcome to Acme")]
    )

    chunks: list[StreamChunk] = []
    iterator = await machine.turn("hello", stream="text")
    async for chunk in iterator:
        chunks.append(chunk)

    assert chunks, "expected at least one chunk"
    assert chunks[-1].done is True
    assert isinstance(chunks[-1].turn, Turn)
    reassembled = "".join(c.text for c in chunks)
    assert reassembled == chunks[-1].turn.text
    assert chunks[-1].turn.metadata["to_node"] == "collect_name"


async def test_stream_bool_true_is_equivalent_to_text() -> None:
    machine = DialogMachine(flow=Flow.load(FIXTURE), llm="openai/gpt-4o-mini")
    # First response is the criteria JSON, second is the new node's generated reply.
    machine._llm = StubProvider(  # type: ignore[assignment]
        [
            _criteria_json(edge_id="greet_to_name", response="ok"),
            "Hello there friend",
        ]
    )
    iterator = await machine.turn("hi", stream=True)
    chunks = [c async for c in iterator]
    assert chunks[-1].done is True
    reassembled = "".join(c.text for c in chunks)
    assert reassembled == chunks[-1].turn.text
    assert reassembled  # non-empty


async def test_stream_done_chunk_carries_turn_metadata() -> None:
    machine = DialogMachine(flow=Flow.load(FIXTURE), llm="openai/gpt-4o-mini")
    machine._llm = StubProvider(  # type: ignore[assignment]
        [_criteria_json(edge_id=None, response="please repeat")]
    )
    iterator = await machine.turn("hi", stream="text")
    chunks = [c async for c in iterator]
    assert chunks[-1].done is True
    assert chunks[-1].turn is not None
    assert chunks[-1].turn.metadata["outcome"] == "stay"
    # only the final chunk carries the Turn; earlier ones do not
    assert all(c.turn is None for c in chunks[:-1])
