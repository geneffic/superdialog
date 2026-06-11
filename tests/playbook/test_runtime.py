import textwrap

from superdialog.playbook.events import (
    ExternalEvent,
    SessionEndEvent,
    SlotWriteEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_toolexec import FakeHttp


def _runtime(llm_payload: dict, http_responses=()) -> PlaybookRuntime:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    return PlaybookRuntime(
        pb,
        director_llm=CannedLLM(llm_payload),
        http=FakeHttp(list(http_responses)),
    )


async def test_session_start_enters_initial_checkpoint_and_seeds_env() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML).model_copy(
        update={"env": {"API_BASE_URL": "https://api.test"}}
    )
    rt = PlaybookRuntime(
        pb,
        director_llm=CannedLLM({"slots": {}, "advance": None, "note": None}),
        http=FakeHttp([]),
    )
    await rt.start()
    assert rt.state.checkpoint_id == "booking.collect"
    assert rt.state.env["API_BASE_URL"] == "https://api.test"


async def test_user_event_advances_through_pipeline_to_terminal() -> None:
    rt = _runtime(
        {
            "slots": {"city": "Pune", "date": "2026-06-11"},
            "advance": "booking.confirm",
            "note": None,
        },
        http_responses=[(200, {"data": {"hold_id": "h1"}})],
    )
    await rt.start()
    speech = await rt.on_user_text("Pune tomorrow")
    # collect -> confirm (llm) -> pipeline (hold ok -> continue) ->
    # expr rule pipeline.ok -> close (terminal)
    assert rt.state.ended and rt.state.outcome == "confirmed"
    assert any(isinstance(e, SessionEndEvent) for e in rt.log.events)
    # confirm's say_verbatim surfaced as pass-through speech + logged utterance
    assert any("held" in s for s in speech)
    assert any(
        e.type == "utterance" and e.role == "assistant" and "held" in e.text
        for e in rt.log.events
    )


async def test_silence_policy_prompts_then_routes() -> None:
    rt = _runtime({"slots": {}, "advance": None, "note": None})
    await rt.start()
    r1 = await rt.on_external(ExternalEvent(kind="silence", name="user_silence"))
    assert r1.prompt == "Can you hear me?"
    r2 = await rt.on_external(ExternalEvent(kind="silence", name="user_silence"))
    assert r2.prompt == "Are you there?"
    r3 = await rt.on_external(ExternalEvent(kind="silence", name="user_silence"))
    assert r3.prompt is None
    assert rt.state.checkpoint_id == "booking.close"


async def test_degraded_director_is_logged_not_fatal() -> None:
    class BadLLM:
        async def complete(self, messages, **kwargs) -> str:
            return "not json {"

    pb = Playbook.from_yaml(MINIMAL_YAML)
    rt = PlaybookRuntime(pb, director_llm=BadLLM(), http=FakeHttp([]))
    await rt.start()
    await rt.on_user_text("hello?")
    assert any(e.type == "degraded" for e in rt.log.events)
    assert not rt.state.ended


async def test_stale_talker_speech_triggers_repair_note() -> None:
    rt = _runtime({"slots": {}, "advance": None, "note": None})
    await rt.start()
    stale_version = rt.state.version
    rt.log.append(UtteranceEvent(role="user", text="my city is Pune"))
    rt.log.append(
        UtteranceEvent(
            role="assistant",
            text="Which city would you like?",
            spoke_from_version=stale_version,
        )
    )
    rt.log.append(
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")
    )
    await rt.check_repairs()
    notes = [e for e in rt.log.events if e.type == "steering_note"]
    assert any(n.kind == "repair" for n in notes)


TURN_BUDGET_YAML = textwrap.dedent("""
    journeys:
      j:
        checkpoints:
          - id: a
            goal: "chat"
            turn_budget: 1
            slots:
              x: {type: str}
            advance_when:
              - {when: "user is done", judge: llm, to: j.b}
          - id: b
            terminal: true
            outcome: done
""")


async def test_turn_budget_steers() -> None:
    pb = Playbook.from_yaml(TURN_BUDGET_YAML)
    rt = PlaybookRuntime(
        pb,
        director_llm=CannedLLM({"slots": {}, "advance": None, "note": None}),
        http=FakeHttp([]),
    )
    await rt.start()
    await rt.on_user_text("hello")
    assert not [e for e in rt.log.events if e.type == "steering_note"]
    await rt.on_user_text("still chatting")
    notes = [e for e in rt.log.events if e.type == "steering_note"]
    assert any("wrap" in n.text for n in notes)
    # budget exceeded but within grace and no on_failure: stay put
    assert rt.state.checkpoint_id == "j.a"
