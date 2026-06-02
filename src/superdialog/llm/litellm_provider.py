"""LitellmProvider — LLMProvider impl backed by litellm.acompletion."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

import litellm

from .provider import CompletionResult, StreamChunk


class LitellmProvider:
    def __init__(self, model: str, **default_opts: Any) -> None:
        self.model = model
        self.default_opts: dict[str, Any] = default_opts

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        merged = {**self.default_opts, **opts}
        t0 = time.perf_counter()
        resp = await litellm.acompletion(
            model=self.model, messages=messages, tools=tools, **merged
        )
        msg = resp.choices[0].message
        raw_calls = msg.tool_calls or []
        tool_calls = [
            tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
            for tc in raw_calls
        ]
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        print(
            f"[LLM] {latency_ms:.0f}ms  "
            f"in={prompt_tokens} out={completion_tokens} tok  "
            f"model={self.model}"
        )
        return CompletionResult(
            text=msg.content or "",
            tool_calls=tool_calls,
            metadata={
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": self.model,
            },
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        merged = {**self.default_opts, **opts, "stream": True}
        resp = await litellm.acompletion(
            model=self.model, messages=messages, tools=tools, **merged
        )
        async for chunk in resp:
            delta = chunk.choices[0].delta
            tcs = getattr(delta, "tool_calls", None)
            tc_delta: dict[str, Any] | None = None
            if tcs:
                first = tcs[0]
                tc_delta = (
                    first.model_dump() if hasattr(first, "model_dump") else dict(first)
                )
            yield StreamChunk(
                text=getattr(delta, "content", None),
                tool_call_delta=tc_delta,
                done=chunk.choices[0].finish_reason is not None,
            )
