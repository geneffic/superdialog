import pytest

from superdialog.chat_context import ChatContext, ChatMessage
from superdialog.session.record import SessionRecord
from superdialog.session.store import InMemorySessionStore, NullSessionStore


def _record() -> SessionRecord:
    return SessionRecord(
        chat_ctx=ChatContext(items=[ChatMessage("user", "hi")]),
        metadata={"user": "alice"},
    )


@pytest.mark.asyncio
async def test_in_memory_store_roundtrip() -> None:
    store = InMemorySessionStore()
    rec = _record()
    await store.save("X", rec)
    loaded = await store.load("X")
    assert loaded == rec


@pytest.mark.asyncio
async def test_in_memory_store_delete() -> None:
    store = InMemorySessionStore()
    await store.save("X", _record())
    await store.delete("X")
    assert await store.load("X") is None


@pytest.mark.asyncio
async def test_in_memory_store_missing_id_returns_none() -> None:
    store = InMemorySessionStore()
    assert await store.load("never-existed") is None


@pytest.mark.asyncio
async def test_null_store_drops_writes() -> None:
    store = NullSessionStore()
    await store.save("X", _record())
    assert await store.load("X") is None


def test_session_record_defaults() -> None:
    rec = SessionRecord()
    assert rec.version == 1
    assert rec.flow_state is None
    assert rec.metadata == {}
    assert rec.chat_ctx.items == []
