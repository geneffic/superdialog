"""PlaybookAgent: the Playbook engine behind the public Agent protocol."""

from superdialog.agent import Agent, TurnResult
from superdialog.playbook import EventLog, PlaybookAgent
from superdialog.playbook.events import UtteranceEvent
from superdialog.playbook.models import Playbook
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

_IDLE_VERDICT: dict = {"slots": {}, "advance": None, "note": None}


def _agent(
    verdict: dict | None = None,
    http_responses: list[tuple[int, dict]] | None = None,
) -> PlaybookAgent:
    return PlaybookAgent(
        playbook=Playbook.from_yaml(MINIMAL_YAML),
        talker_llm=StreamLLM(["Which", " city?"]),
        director_llm=CannedLLM(verdict or _IDLE_VERDICT),
        http=FakeHttp(http_responses or []),
    )


def test_satisfies_agent_protocol() -> None:
    assert isinstance(_agent(), Agent)


async def test_turn_returns_text_and_metadata() -> None:
    agent = _agent()
    result = await agent.turn("hello")
    assert isinstance(result, TurnResult)
    assert result.text == "Which city?"
    assert result.metadata["checkpoint"] == "booking.collect"


async def test_streaming_turn_yields_real_chunks() -> None:
    agent = _agent()
    result = await agent.turn("hello", stream=True)
    assert not isinstance(result, TurnResult)
    chunks = [c async for c in result]
    assert [c.text for c in chunks if c.text] == ["Which", " city?"]
    assert chunks[-1].done


async def test_chat_ctx_round_trip() -> None:
    agent = _agent()
    await agent.turn("hello")
    ctx = agent.chat_ctx
    agent2 = _agent()
    agent2.load_chat_ctx(ctx)
    assert agent2.chat_ctx.items == ctx.items


def test_assist_logs_system_message() -> None:
    agent = _agent()
    agent.assist("note")
    assert any(
        e.role == "system" and e.text == "note" for e in agent.runtime.state.transcript
    )


async def test_talker_speech_logged() -> None:
    agent = _agent()
    await agent.turn("hello")
    spoken = [
        e
        for e in agent.runtime.log.events
        if isinstance(e, UtteranceEvent)
        and e.role == "assistant"
        and e.text == "Which city?"
    ]
    assert len(spoken) == 1  # exactly once — never duplicated
    assert spoken[0].spoke_from_version is not None


async def test_event_log_round_trip() -> None:
    agent = _agent()
    await agent.turn("hello")
    restored = EventLog.from_jsonl(agent.event_log.to_jsonl())
    assert restored.version == agent.event_log.version
    agent2 = _agent()
    agent2.load_event_log(restored)
    assert agent2.runtime.state.checkpoint_id == agent.runtime.state.checkpoint_id


async def test_turn_includes_pass_through() -> None:
    agent = _agent(
        verdict={
            "slots": {"city": "Pune", "date": "2026-06-12"},
            "advance": "booking.confirm",
            "note": None,
        },
        http_responses=[(200, {"data": {"hold_id": "h1"}})],
    )
    result = await agent.turn("Pune tomorrow")
    assert isinstance(result, TurnResult)
    assert "Which city?" in result.text
    assert "held" in result.text
    assert result.metadata["ended"] is True
    assert result.metadata["outcome"] == "confirmed"
