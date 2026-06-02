"""SessionStore Protocol + default implementations.

External backends (Redis, file, SQLite) live in follow-up changes and are
opt-in via package extras. The two implementations here cover the common
in-process cases: persistent-for-process-lifetime (``InMemorySessionStore``)
and "do not persist" (``NullSessionStore``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .record import SessionRecord


@runtime_checkable
class SessionStore(Protocol):
    """Persistence backend for :class:`SessionRecord` instances."""

    async def load(self, session_id: str) -> SessionRecord | None: ...

    async def save(self, session_id: str, record: SessionRecord) -> None: ...

    async def delete(self, session_id: str) -> None: ...


class InMemorySessionStore:
    """Dict-backed store. Persistent for the process lifetime only."""

    def __init__(self) -> None:
        self._records: dict[str, SessionRecord] = {}

    async def load(self, session_id: str) -> SessionRecord | None:
        return self._records.get(session_id)

    async def save(self, session_id: str, record: SessionRecord) -> None:
        self._records[session_id] = record

    async def delete(self, session_id: str) -> None:
        self._records.pop(session_id, None)


class NullSessionStore:
    """No-op store: drops every save, returns ``None`` on every load.

    Use when the worker should not persist state (voice calls, ephemeral
    sessions, tests). Distinct from ``InMemorySessionStore`` which DOES
    persist across acquires within the process.
    """

    async def load(self, session_id: str) -> SessionRecord | None:
        return None

    async def save(self, session_id: str, record: SessionRecord) -> None:
        return None

    async def delete(self, session_id: str) -> None:
        return None


__all__ = ["InMemorySessionStore", "NullSessionStore", "SessionStore"]
