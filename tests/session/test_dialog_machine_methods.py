"""Group 2: DialogMachine view/load/assist additions."""

from __future__ import annotations

import warnings

import pytest

from superdialog.chat_context import ChatContext, ChatMessage
from superdialog.flow.models import ConversationFlow, FlowNode
from superdialog.machine.machine import DialogStateMachine


class _FakeAdapter:
    """Minimal adapter stub: only needs to exist for DM construction."""

    async def deliver_speech(self, *args, **kwargs):
        pass


def _flow() -> ConversationFlow:
    return ConversationFlow(
        id="t",
        initial_node="start",
        system_prompt="test",
        nodes=[
            FlowNode(id="start", name="Start", static_text="Hi.", is_final=True),
        ],
    )


@pytest.mark.asyncio
async def test_chat_ctx_reflects_history() -> None:
    dm = await DialogStateMachine.from_flow(flow=_flow(), adapter=_FakeAdapter())
    dm.context.add_user_message("hello")
    ctx = dm.chat_ctx
    assert any(m.role == "user" and m.content == "hello" for m in ctx.items)


@pytest.mark.asyncio
async def test_load_chat_ctx_replaces_history() -> None:
    dm = await DialogStateMachine.from_flow(flow=_flow(), adapter=_FakeAdapter())
    dm.load_chat_ctx(
        ChatContext(items=[ChatMessage("user", "foo"), ChatMessage("assistant", "bar")])
    )
    assert dm.context.data.history == [
        {"role": "user", "content": "foo"},
        {"role": "assistant", "content": "bar"},
    ]


@pytest.mark.asyncio
async def test_flow_state_round_trip() -> None:
    dm = await DialogStateMachine.from_flow(flow=_flow(), adapter=_FakeAdapter())
    dm.context.current_node_id = "start"
    dm.context.userdata = {"name": "Alice"}
    snapshot = dm.flow_state
    assert snapshot.current_node_id == "start"
    assert snapshot.userdata == {"name": "Alice"}

    dm2 = await DialogStateMachine.from_flow(flow=_flow(), adapter=_FakeAdapter())
    dm2.load_flow_state(snapshot)
    assert dm2.context.current_node_id == "start"
    assert dm2.context.userdata == {"name": "Alice"}
    assert dm2.state == "start"


@pytest.mark.asyncio
async def test_assist_appends_system_message() -> None:
    dm = await DialogStateMachine.from_flow(flow=_flow(), adapter=_FakeAdapter())
    dm.assist("Be brief.")
    last = dm.context.data.history[-1]
    assert last == {"role": "system", "content": "Be brief."}


@pytest.mark.asyncio
async def test_inject_system_is_deprecated_alias() -> None:
    dm = await DialogStateMachine.from_flow(flow=_flow(), adapter=_FakeAdapter())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dm.inject_system("Be brief.")
        assert any(issubclass(item.category, DeprecationWarning) for item in w)
    assert dm.context.data.history[-1] == {"role": "system", "content": "Be brief."}


@pytest.mark.asyncio
async def test_assist_with_empty_text_is_noop() -> None:
    dm = await DialogStateMachine.from_flow(flow=_flow(), adapter=_FakeAdapter())
    before = list(dm.context.data.history)
    dm.assist("")
    assert dm.context.data.history == before
