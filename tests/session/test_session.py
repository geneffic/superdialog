from superdialog.chat_context import ChatContext, ChatMessage
from superdialog.session.session import Session, SessionHandle


class _StubAgent:
    def __init__(self) -> None:
        self.assist_calls: list[str] = []
        self._chat = ChatContext()

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat

    def load_chat_ctx(self, ctx: ChatContext) -> None:
        self._chat = ctx

    def assist(self, text: str) -> None:
        self.assist_calls.append(text)

    async def turn(self, text: str, *, stream: bool = False):
        return {"text": f"echo: {text}"}


def test_session_assist_appends_system_message() -> None:
    s = Session(id="X")
    s.assist("be brief")
    assert s.chat_ctx.items == [ChatMessage("system", "be brief")]


def test_session_assist_empty_text_noop() -> None:
    s = Session(id="X")
    s.assist("")
    assert s.chat_ctx.items == []


def test_session_handle_state_exposes_session_fields() -> None:
    s = Session(id="X")
    s.chat_ctx.items.append(ChatMessage("user", "hi"))
    h = SessionHandle(s, _StubAgent())
    state = h.state
    assert state["id"] == "X"
    assert state["chat_ctx"] is s.chat_ctx
    assert state["flow_state"] is None


def test_session_handle_assist_propagates_to_session_and_agent() -> None:
    s = Session(id="X")
    agent = _StubAgent()
    h = SessionHandle(s, agent)
    h.assist("note")
    assert s.chat_ctx.items == [ChatMessage("system", "note")]
    assert agent.assist_calls == ["note"]


async def test_session_handle_turn_delegates_to_agent() -> None:
    s = Session(id="X")
    agent = _StubAgent()
    h = SessionHandle(s, agent)
    result = await h.turn("hi")
    assert result == {"text": "echo: hi"}
