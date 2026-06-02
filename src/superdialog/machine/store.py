"""ContextStore -- pluggable persistence for FlowContext."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from superdialog.machine.models import FlowContext


@runtime_checkable
class ContextStore(Protocol):
    """Contract for persisting FlowContext across sessions."""

    async def save(self, session_id: str, context: FlowContext) -> None: ...

    async def load(self, session_id: str) -> FlowContext | None: ...

    async def delete(self, session_id: str) -> None: ...


class InMemoryContextStore:
    """In-memory ContextStore backed by a dict."""

    def __init__(self) -> None:
        self._store: dict[str, FlowContext] = {}

    async def save(self, session_id: str, context: FlowContext) -> None:
        self._store[session_id] = context

    async def load(self, session_id: str) -> FlowContext | None:
        return self._store.get(session_id)

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
