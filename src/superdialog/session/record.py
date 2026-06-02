"""SessionRecord — the durable wire format SessionStores read and write."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..chat_context import ChatContext
from ..flow_state import FlowState


@dataclass
class SessionRecord:
    """Versioned snapshot persisted by :class:`SessionStore` implementations.

    ``flow_state`` is ``None`` for non-DialogMachine agents — only the
    chat history is durable for those brains.
    """

    chat_ctx: ChatContext = field(default_factory=ChatContext)
    flow_state: FlowState | None = None
    version: int = 1
    created_at: float = field(default_factory=time.time)
    last_turn_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["SessionRecord"]
