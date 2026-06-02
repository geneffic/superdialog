"""SessionWorker — process-level session multiplexer.

Multiplexes N concurrent sessions inside one process, each backed by its own
``Agent`` instance created via the configured ``agent_factory`` closure.
Persistence runs through a pluggable :class:`SessionStore`; concurrency on a
single session_id is serialised via a pluggable :class:`LockBackend`.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator, Callable

from .lock import AsyncioLockBackend, LockBackend
from .record import SessionRecord
from .session import Session, SessionHandle
from .store import InMemorySessionStore, SessionStore

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)

DEFAULT_MAX_SESSIONS = 1000


class SessionWorker:
    """Multiplexes Sessions backed by per-session Agent instances.

    The Worker is the only object callers need to share across handlers. It
    builds a fresh Agent per session via ``agent_factory`` and caches the
    (Session, Agent) pair until it is evicted by the LRU policy or
    explicitly closed.
    """

    def __init__(
        self,
        *,
        agent_factory: Callable[[], "Agent"],
        store: SessionStore | None = None,
        lock_backend: LockBackend | None = None,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        self._agent_factory = agent_factory
        self._store: SessionStore = store or InMemorySessionStore()
        self._lock_backend: LockBackend = lock_backend or AsyncioLockBackend()
        self._max_sessions = max_sessions
        self._cache: "OrderedDict[str, tuple[Session, Agent]]" = OrderedDict()

    @asynccontextmanager
    async def acquire(self, session_id: str) -> AsyncIterator[SessionHandle]:
        """Acquire exclusive access to a session by id.

        On entry: lock(session_id), load or create the Session+Agent pair,
        load any persisted state into the Agent, yield a SessionHandle.
        On exit: pull updated state out of the Agent, persist a SessionRecord,
        keep the pair in the LRU cache for the next acquire.
        """
        async with self._lock_backend.acquire(session_id):
            session, agent = await self._load_or_create(session_id)
            try:
                yield SessionHandle(session, agent)
            finally:
                self._sync_session_from_agent(session, agent)
                await self._persist(session)

    async def close_session(self, session_id: str) -> None:
        """Evict a session explicitly: flush state, drop the cache entry."""
        async with self._lock_backend.acquire(session_id):
            pair = self._cache.pop(session_id, None)
            if pair is None:
                return
            session, agent = pair
            self._sync_session_from_agent(session, agent)
            await self._persist(session)

    # ---- internals --------------------------------------------------------

    async def _load_or_create(self, session_id: str) -> "tuple[Session, Agent]":
        cached = self._cache.get(session_id)
        if cached is not None:
            self._cache.move_to_end(session_id)
            return cached

        # Construct fresh Agent + load any persisted record
        agent = self._agent_factory()
        record = await self._store.load(session_id)
        if record is not None:
            agent.load_chat_ctx(record.chat_ctx)
            if record.flow_state is not None and hasattr(agent, "load_flow_state"):
                agent.load_flow_state(record.flow_state)
            session = Session(
                id=session_id,
                chat_ctx=record.chat_ctx,
                flow_state=record.flow_state,
                metadata=dict(record.metadata),
            )
        else:
            session = Session(id=session_id)

        self._cache[session_id] = (session, agent)
        await self._enforce_lru()
        return session, agent

    def _sync_session_from_agent(self, session: Session, agent: "Agent") -> None:
        """Pull updated state out of the Agent back into the Session."""
        session.chat_ctx = agent.chat_ctx
        flow_state = getattr(agent, "flow_state", None)
        if flow_state is not None:
            session.flow_state = flow_state

    async def _persist(self, session: Session) -> None:
        record = SessionRecord(
            chat_ctx=session.chat_ctx,
            flow_state=session.flow_state,
            last_turn_at=time.time(),
            metadata=dict(session.metadata),
        )
        await self._store.save(session.id, record)

    async def _enforce_lru(self) -> None:
        while len(self._cache) > self._max_sessions:
            evicted_id, pair = self._cache.popitem(last=False)
            session, agent = pair
            self._sync_session_from_agent(session, agent)
            await self._persist(session)
            logger.debug("[session] evicted %s from cache (LRU full)", evicted_id)

    @property
    def cache_size(self) -> int:
        return len(self._cache)


__all__ = ["SessionWorker"]
