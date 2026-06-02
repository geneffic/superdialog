"""LiveKit adapter — exercise the stream against a fake chat context.

We do not depend on a real LiveKit Agent in this test; we drive
``DialogMachineStream`` directly with a stand-in chat context and assert
on the chunks it yields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("livekit.agents")

from superdialog.adapters import livekit as lk_adapter  # noqa: E402


@dataclass
class _Msg:
    role: str
    content: str


class _ChatCtx:
    def __init__(self, messages: list[_Msg]) -> None:
        self.messages = messages


@pytest.mark.asyncio
async def test_stream_extracts_user_text_and_emits_chunks(fake_dm) -> None:
    ctx = _ChatCtx([_Msg("system", "sys"), _Msg("user", "ping")])
    stream = lk_adapter.DialogMachineStream(fake_dm, chat_ctx=ctx)
    collected: list[Any] = [chunk async for chunk in stream]
    assert collected, "expected at least one chunk"
    assert fake_dm.received == ["ping"]


@pytest.mark.asyncio
async def test_stream_handles_empty_context(fake_dm) -> None:
    stream = lk_adapter.DialogMachineStream(fake_dm, chat_ctx=None)
    chunks = [c async for c in stream]
    assert chunks  # falls through with empty user text
    assert fake_dm.received == [""]


def test_extract_latest_user_text_prefers_last_user_msg() -> None:
    ctx = _ChatCtx(
        [_Msg("user", "first"), _Msg("assistant", "a"), _Msg("user", "last")]
    )
    assert lk_adapter._extract_latest_user_text(ctx) == "last"


def test_extract_latest_user_text_handles_dict_messages() -> None:
    ctx = _ChatCtx(
        [
            {"role": "system", "content": "ignored"},  # type: ignore[list-item]
            {"role": "user", "content": "hello"},  # type: ignore[list-item]
        ]
    )
    assert lk_adapter._extract_latest_user_text(ctx) == "hello"


def test_extract_latest_user_text_returns_empty_when_no_user() -> None:
    assert lk_adapter._extract_latest_user_text(_ChatCtx([])) == ""
