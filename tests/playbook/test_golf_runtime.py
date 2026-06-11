"""Golf flow smoke: the compiled playbook runs end-to-end through the engine.

This is a SMOKE CONTRACT, not a behavior oracle: it asserts the compiled
artifact runs without crashing and the key mechanics hold (preload tools,
env seeding, intermediate pipelines, silence policy, log round-trip).
Behavioral parity with the legacy machine is a deferred differential eval.
"""

import json
from pathlib import Path
from typing import Any

from superdialog.flow.models import ConversationFlow
from superdialog.playbook import EventLog, PlaybookAgent
from superdialog.playbook.compiler import compile_flow
from superdialog.playbook.events import (
    AdvanceEvent,
    EnvWriteEvent,
    ExternalEvent,
    SlotWriteEvent,
    ToolCallEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

GOLF = Path(__file__).parents[1] / "fixtures" / "flow" / "golf_booking.json"


def _pb() -> Playbook:
    flow = ConversationFlow.model_validate(json.loads(GOLF.read_text()))
    return compile_flow(flow)


class SequencedLLM:
    """Director LLM: one canned verdict per call, then repeats the last."""

    def __init__(self, verdicts: list[dict]) -> None:
        self.verdicts = list(verdicts)
        self.calls = 0

    async def complete(self, messages: list[dict], **kwargs: object) -> str:
        i = min(self.calls, len(self.verdicts) - 1)
        self.calls += 1
        return json.dumps(self.verdicts[i])


class TolerantFakeHttp(FakeHttp):
    """FakeHttp that serves (200, {}) once the scripted queue runs out."""

    async def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: dict,
        body: dict | None,
        timeout: float,
    ) -> tuple[int, dict]:
        if not self.responses:
            self.responses.append((200, {}))
        return await super().__call__(
            method=method, url=url, headers=headers, body=body, timeout=timeout
        )


def _assert_calls_paired(log: EventLog) -> None:
    """Every ToolCallEvent has a later ToolResultEvent for the same tool."""
    pending: list[str] = []
    for e in log.events:
        if isinstance(e, ToolCallEvent):
            pending.append(e.tool)
        elif isinstance(e, ToolResultEvent) and e.tool in pending:
            pending.remove(e.tool)  # earliest pending call with this name
    assert pending == [], f"unmatched tool calls: {pending}"


async def test_golf_session_smoke() -> None:
    pb = _pb()
    # Discover rule targets from the compiled artifact — never hardcode
    # synthesized intermediate ids.
    greeting = pb.checkpoint(pb.initial_checkpoint_id)
    book_rule = next(r for r in greeting.advance_when if "collect_booking" in r.to)
    collect = pb.checkpoint(book_rule.to)
    needed = {"date", "preferred_time", "booking_players", "course_id"}
    avail_rule = next(r for r in collect.advance_when if needed <= set(r.requires))
    present = pb.checkpoint("main.present_available_slot")
    confirm_rule = next(
        r
        for r in present.advance_when
        if r.to.endswith("collect_booking_confirmation_details")
    )
    money_targets = {
        r.to
        for r in pb.checkpoint(confirm_rule.to).advance_when
        if "__" in r.to  # synthesized money intermediate(s)
    }
    slot_values = {
        "date": "2026-06-12",
        "preferred_time": "morning",
        "booking_players": "4",
        "course_id": "course-oxford",
    }
    verdicts = [
        {"slots": {}, "advance": book_rule.to, "note": None},
        {
            "slots": {k: slot_values.get(k, "x") for k in avail_rule.requires},
            "advance": avail_rule.to,
            "note": None,
        },
        {
            "slots": {k: "s-1" for k in confirm_rule.requires},
            "advance": confirm_rule.to,
            "note": None,
        },
    ]
    # Greeting's 5 preload tools fire during start(): their responses come
    # FIRST. auth seeds ACCESS_TOKEN; players-search seeds player_id, which
    # un-skips players-get/bookings (`when: env.player_id`).
    http = TolerantFakeHttp(
        [
            (200, {"data": {"access_token": "tok-1"}}),
            (200, {"data": {"player_id": "p-1"}}),
            (200, {"data": {"name": "Asha"}}),
            (200, {"data": {"bookings": []}}),
            (200, {"data": {"courses": [{"course_id": "course-oxford"}]}}),
            (200, {"data": {"slots": [{"slot_id": "s-1", "time": "07:00"}]}}),
        ]
        + [(200, {})] * 14
    )
    agent = PlaybookAgent(
        playbook=pb,
        talker_llm=StreamLLM(["Okay!"]),
        director_llm=SequencedLLM(verdicts),
        http=http,
    )
    await agent.runtime.start()
    for text in (
        "I want to book a tee time",
        "Pune tomorrow morning 4 players at Oxford course",
        "yes go ahead",
    ):
        await agent.turn(text)  # smoke: no exception across the session

    log = agent.runtime.log
    moves = [e for e in log.events if isinstance(e, AdvanceEvent) and e.rule != "init"]
    assert len(moves) >= 2  # checkpoint changed at least twice
    _assert_calls_paired(log)
    # env seeding proven: recorded tool urls start with the fixture base url
    base = pb.env["API_BASE_URL"]
    assert base.startswith("https://api.teetime.golfai.in")
    urls = [e.args["url"] for e in log.events if isinstance(e, ToolCallEvent)]
    assert any(u.startswith(base) for u in urls)
    restored = EventLog.from_jsonl(log.to_jsonl())
    assert restored.version == log.version
    # Don't force the money pipeline; assert only if the session reached it.
    if any(e.to_checkpoint in money_targets for e in moves):
        called = " ".join(c["url"] for c in http.calls)
        assert "/slots/hold" in called and "/bookings/confirm" in called
    print("final checkpoint:", agent.runtime.state.checkpoint_id)


async def test_golf_money_pipeline_directly() -> None:
    """Deterministic: park at the confirm step, advance into the money chain."""
    pb = _pb()
    confirm_ref = "main.collect_booking_confirmation_details"
    money_rule = next(
        r for r in pb.checkpoint(confirm_ref).advance_when if "__" in r.to
    )
    ok_rule = next(
        r for r in pb.checkpoint(money_rule.to).advance_when if r.when == "pipeline.ok"
    )
    http = FakeHttp(
        [
            (200, {"data": {"hold_id": "h1"}}),
            (200, {"data": {"booking_id": "b1"}}),
        ]
    )
    rt = PlaybookRuntime(
        pb,
        director_llm=CannedLLM({"slots": {}, "advance": money_rule.to, "note": None}),
        http=http,
    )
    # Park state: seed env, enter the confirm checkpoint, confirm requires.
    for key, value in pb.env.items():
        rt.log.append(EnvWriteEvent(key=key, value=value))
    rt.log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint=confirm_ref, rule="init")
    )
    seed: dict[str, Any] = {
        "slot_id": "s-1",
        "players_array": [{"name": "Asha", "player_id": "p-1"}],
        "addons_array": [],
        "special_requests": "none",
    }
    for k in money_rule.requires:
        rt.log.append(
            SlotWriteEvent(
                key=k, value=seed.get(k, "x"), status="confirmed", by="director"
            )
        )

    pass_through = await rt.on_user_text("yes confirm")

    assert rt.state.checkpoint_id == ok_rule.to  # main.booking_close
    assert rt.state.ended and rt.state.outcome == "booking_close"
    # hold AND confirm were both called, in order
    assert http.calls[0]["url"].endswith("/slots/hold")
    assert http.calls[1]["url"].endswith("/bookings/confirm")
    assert http.calls[1]["body"]["hold_id"] == "h1"  # env threaded hold→confirm
    spoken = pass_through + [
        e.text
        for e in rt.log.events
        if isinstance(e, UtteranceEvent) and e.role == "assistant"
    ]
    assert any("confirmed" in t for t in spoken)  # booking_close verbatim


async def test_golf_silence_policy() -> None:
    pb = _pb()
    policy = pb.policies.silence
    assert policy is not None
    rt = PlaybookRuntime(
        pb,
        director_llm=CannedLLM({"slots": {}, "advance": None, "note": None}),
        http=TolerantFakeHttp([]),
    )
    await rt.start()
    results = [
        await rt.on_external(ExternalEvent(kind="silence", name="user_silent"))
        for _ in range(3)
    ]
    assert [r.prompt for r in results[:2]] == policy.prompts  # prompts first
    assert results[2].prompt is None  # third silence routes instead
    assert rt.state.checkpoint_id == policy.then  # main.call_end
    assert rt.state.ended
