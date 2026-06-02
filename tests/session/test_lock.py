import asyncio

import pytest

from superdialog.session.lock import AsyncioLockBackend


@pytest.mark.asyncio
async def test_same_key_serialises() -> None:
    backend = AsyncioLockBackend()
    order: list[str] = []

    async def task(name: str, delay: float) -> None:
        async with backend.acquire("X"):
            order.append(f"start-{name}")
            await asyncio.sleep(delay)
            order.append(f"end-{name}")

    await asyncio.gather(task("a", 0.02), task("b", 0.01))
    # Task a must fully finish before task b starts (or vice-versa).
    assert order in (
        ["start-a", "end-a", "start-b", "end-b"],
        ["start-b", "end-b", "start-a", "end-a"],
    )


@pytest.mark.asyncio
async def test_different_keys_run_in_parallel() -> None:
    backend = AsyncioLockBackend()
    timeline: list[tuple[str, str]] = []

    async def task(name: str, key: str) -> None:
        async with backend.acquire(key):
            timeline.append(("enter", name))
            await asyncio.sleep(0.02)
            timeline.append(("exit", name))

    await asyncio.gather(task("a", "X"), task("b", "Y"))
    # Both should enter before either exits if they run in parallel.
    enter_events = [t for t in timeline if t[0] == "enter"]
    exit_events = [t for t in timeline if t[0] == "exit"]
    assert len(enter_events) == 2
    assert timeline.index(enter_events[1]) < timeline.index(exit_events[0])
