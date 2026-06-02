"""Public turn-result types returned by :class:`DialogMachine`.

Spec reference: ``docs/02-api-reference.md`` (``Turn``, ``ToolCall``, the
``stream="text"`` async-iterator contract). The streaming variant emits
:class:`StreamChunk` items; the synchronous variant returns a single
:class:`Turn`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM during a turn."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    """Complete turn result returned by ``DialogMachine.turn`` (non-streaming)."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """One incremental chunk yielded by ``DialogMachine.turn(stream="text")``.

    ``done`` is ``True`` only on the final chunk; that chunk also carries
    the assembled ``Turn`` for callers that need the full result without
    re-aggregating tokens.
    """

    text: str = ""
    done: bool = False
    turn: Turn | None = None
