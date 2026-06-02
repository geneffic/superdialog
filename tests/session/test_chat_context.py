from dataclasses import asdict

from superdialog.chat_context import ChatContext, ChatMessage


def test_empty_chat_context() -> None:
    ctx = ChatContext()
    assert ctx.items == []


def test_append_chat_message() -> None:
    ctx = ChatContext()
    ctx.append(ChatMessage(role="user", content="hi"))
    assert len(ctx.items) == 1
    assert ctx.items[0].role == "user"
    assert ctx.items[0].content == "hi"


def test_copy_returns_independent_instance() -> None:
    ctx = ChatContext(items=[ChatMessage("user", "hi")])
    dup = ctx.copy()
    dup.append(ChatMessage("assistant", "hello"))
    assert len(ctx.items) == 1
    assert len(dup.items) == 2


def test_dataclass_asdict_roundtrip() -> None:
    ctx = ChatContext(
        items=[
            ChatMessage("system", "be brief"),
            ChatMessage("user", "ping"),
            ChatMessage("assistant", "pong"),
        ]
    )
    payload = asdict(ctx)
    rebuilt = ChatContext(items=[ChatMessage(**m) for m in payload["items"]])
    assert rebuilt == ctx
