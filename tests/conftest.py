"""Top-level test fixtures shared across the superdialog test suite."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from superdialog.llm.provider import CompletionResult, StreamChunk


class FakeLLMProvider:
    """Scriptable :class:`superdialog.llm.provider.LLMProvider`.

    Records every call into ``.calls`` and pops scripted responses off
    ``.scripted`` in order. When the script is exhausted it returns an
    empty :class:`CompletionResult` so tests that don't care about the
    response still progress.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.scripted: list[CompletionResult] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        self.calls.append({"messages": messages, "tools": tools, "opts": opts})
        if self.scripted:
            return self.scripted.pop(0)
        return CompletionResult(text="", tool_calls=[], metadata={})

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append({"messages": messages, "tools": tools, "opts": opts})
        result = (
            self.scripted.pop(0)
            if self.scripted
            else CompletionResult(text="", tool_calls=[], metadata={})
        )
        yield StreamChunk(text=result.text, tool_call_delta=None, done=True)


@pytest.fixture()
def fake_llm_provider() -> FakeLLMProvider:
    """Returns a fresh :class:`FakeLLMProvider` per test."""
    return FakeLLMProvider()
