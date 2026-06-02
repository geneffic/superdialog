"""PipeCat adapter — verify the processor reacts to TextFrames.

Skips when ``pipecat`` is not installed. Uses PipeCat's real
``FrameProcessor`` so the test exercises the actual subclass we build.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pipecat")

from superdialog.adapters.pipecat import make_processor  # noqa: E402


@pytest.mark.asyncio
async def test_text_frame_drives_turn(fake_dm, monkeypatch: pytest.MonkeyPatch) -> None:
    from pipecat.frames.frames import TextFrame  # type: ignore

    processor = make_processor(fake_dm)
    pushed: list[Any] = []

    async def fake_push(frame: Any, direction: Any = None) -> None:
        pushed.append(frame)

    monkeypatch.setattr(processor, "push_frame", fake_push, raising=False)

    await processor.process_frame(TextFrame("hello"))
    assert fake_dm.received == ["hello"]
    assert pushed, "expected processor to push a reply frame"
    out = pushed[-1]
    assert getattr(out, "text", "") == "hello world"


@pytest.mark.asyncio
async def test_non_text_frame_is_ignored(
    fake_dm, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = make_processor(fake_dm)
    pushed: list[Any] = []

    async def fake_push(frame: Any, direction: Any = None) -> None:
        pushed.append(frame)

    monkeypatch.setattr(processor, "push_frame", fake_push, raising=False)

    class Bogus:
        pass

    await processor.process_frame(Bogus())
    assert fake_dm.received == []
    assert pushed == []
