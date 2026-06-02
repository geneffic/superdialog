"""LockBackend Protocol + default in-process implementation.

A LockBackend serialises concurrent accesses keyed by ``session_id``. The
default (:class:`AsyncioLockBackend`) is in-process; distributed backends
(Redis, etc.) live in follow-up changes.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class LockBackend(Protocol):
    """Per-key async lock surface."""

    def acquire(self, key: str) -> "AsyncContextManagerLock": ...


class AsyncContextManagerLock(Protocol):
    """The object returned by ``LockBackend.acquire`` — an async cm."""

    async def __aenter__(self) -> None: ...

    async def __aexit__(self, *exc_info) -> None: ...


class AsyncioLockBackend:
    """Default in-process backend. Maintains a dict of asyncio.Lock per key."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._table_lock = asyncio.Lock()

    def _get_lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncIterator[None]:
        async with self._table_lock:
            lock = self._get_lock(key)
        async with lock:
            yield


__all__ = ["AsyncioLockBackend", "LockBackend"]
