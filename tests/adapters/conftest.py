"""Test helpers for adapter suites: a fake DialogMachine that does not call out."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from superdialog.stream import StreamChunk, Turn


class FakeDialogMachine:
    """Duck-typed stand-in for :class:`superdialog.DialogMachine`.

    ``turn()`` returns a canned :class:`Turn`; ``turn(stream=True)``
    yields the reply as space-delimited :class:`StreamChunk` items so
    adapters can be exercised without a real LLM.
    """

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.reset_calls = 0
        self.received: list[str] = []
        self.assist_calls: list[str] = []

    def assist(self, text: str) -> None:
        self.assist_calls.append(text)

    async def turn(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> Any:
        self.received.append(text)
        turn = Turn(
            text=self.reply,
            tool_calls=[],
            metadata={"from_node": "n", "to_node": "n", "model": "fake"},
        )
        if not stream:
            return turn

        async def _gen() -> AsyncIterator[StreamChunk]:
            pieces = self.reply.split(" ")
            last = len(pieces) - 1
            for idx, piece in enumerate(pieces):
                yield StreamChunk(
                    text=piece if idx == last else f"{piece} ",
                    done=idx == last,
                    turn=turn if idx == last else None,
                )

        return _gen()

    def reset(self) -> None:
        self.reset_calls += 1


@pytest.fixture()
def fake_dm() -> FakeDialogMachine:
    return FakeDialogMachine(reply="hello world")
