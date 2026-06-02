"""LiveKit-aligned chat history primitives.

Two dataclasses, kept deliberately small. ``ChatMessage`` mirrors LiveKit
Agents' message shape; ``ChatContext`` is the brain-agnostic wire format
for conversation history that flows between Sessions, Stores, and Agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class ChatContext:
    items: list[ChatMessage] = field(default_factory=list)

    def append(self, msg: ChatMessage) -> None:
        self.items.append(msg)

    def copy(self) -> "ChatContext":
        return ChatContext(items=[ChatMessage(m.role, m.content) for m in self.items])


__all__ = ["ChatContext", "ChatMessage", "Role"]
