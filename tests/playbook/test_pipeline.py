from superdialog.playbook.events import EventLog, SlotWriteEvent
from superdialog.playbook.models import (
    MiddlewareSpec,
    PipelineSpec,
    PipelineStep,
    Playbook,
    RetrySpec,
    ToolSpec,
)
from superdialog.playbook.pipeline import PipelineRunner
from superdialog.playbook.state import ConversationState
from superdialog.playbook.toolexec import ToolExecutor
from tests.playbook.test_toolexec import FakeHttp


def _pb(
    steps: list[PipelineStep], middleware: MiddlewareSpec | None = None
) -> Playbook:
    return Playbook(
        journeys={
            "j": {
                "checkpoints": [
                    {"id": "a"},
                    {"id": "fallback"},
                    {"id": "done", "terminal": True},
                ]
            }
        },
        tools=[
            ToolSpec(
                id="hold",
                method="POST",
                url="http://t/hold",
                headers={"Authorization": "Bearer {{ env.ACCESS_TOKEN }}"},
                store_response_as="hold_result",
            ),
            ToolSpec(
                id="confirm",
                method="POST",
                url="http://t/confirm",
                store_response_as="confirm_result",
            ),
            ToolSpec(
                id="refresh",
                method="POST",
                url="http://t/auth",
                env_updates={"ACCESS_TOKEN": "token"},
            ),
        ],
        pipelines=[PipelineSpec(id="p", steps=steps)],
        middleware=middleware,
    )


def _state() -> ConversationState:
    log = EventLog()
    log.append(
        SlotWriteEvent(key="slot_id", value="s1", status="confirmed", by="director")
    )
    state = ConversationState.fold(log)
    state.env["ACCESS_TOKEN"] = "tok-OLD"
    return state


async def test_happy_path_runs_all_steps() -> None:
    pb = _pb(
        [
            PipelineStep(tool="hold", on={"ok": "continue"}),
            PipelineStep(tool="confirm", on={"ok": "j.done"}),
        ]
    )
    runner = PipelineRunner(pb, ToolExecutor(http=FakeHttp([(200, {}), (200, {})])))
    result = await runner.run("p", _state())
    assert result.advance_to == "j.done" and result.ok


async def test_http_code_branch() -> None:
    pb = _pb([PipelineStep(tool="hold", on={"ok": "j.done", "http_409": "j.fallback"})])
    runner = PipelineRunner(
        pb, ToolExecutor(http=FakeHttp([(409, {"error": "taken"})]))
    )
    result = await runner.run("p", _state())
    assert result.advance_to == "j.fallback" and not result.ok


async def test_retry_then_exhaust() -> None:
    pb = _pb(
        [
            PipelineStep(
                tool="confirm",
                on={
                    "ok": "j.done",
                    "failed": RetrySpec(retry=1, on_exhaust="j.fallback"),
                },
            )
        ]
    )
    http = FakeHttp([(503, {}), (503, {})])
    runner = PipelineRunner(pb, ToolExecutor(http=http))
    result = await runner.run("p", _state())
    assert len(http.calls) == 2  # original + 1 retry
    assert result.advance_to == "j.fallback"
    assert result.error_slot == {"error_context": "p:confirm"}


async def test_middleware_refresh_and_replay_on_401() -> None:
    pb = _pb(
        [PipelineStep(tool="hold", on={"ok": "j.done"})],
        middleware=MiddlewareSpec(on_status=401, refresh_with="refresh"),
    )
    http = FakeHttp([(401, {}), (200, {"token": "tok-NEW"}), (200, {})])
    runner = PipelineRunner(pb, ToolExecutor(http=http))
    result = await runner.run("p", _state())
    assert result.advance_to == "j.done"
    assert [c["url"] for c in http.calls] == [
        "http://t/hold",
        "http://t/auth",
        "http://t/hold",
    ]
    # the replayed call must render with the REFRESHED token
    assert http.calls[2]["headers"]["Authorization"] == "Bearer tok-NEW"
    assert http.calls[0]["headers"]["Authorization"] == "Bearer tok-OLD"


async def test_middleware_does_not_loop() -> None:
    pb = _pb(
        [PipelineStep(tool="hold", on={"ok": "j.done"})],
        middleware=MiddlewareSpec(on_status=401, refresh_with="refresh"),
    )
    http = FakeHttp([(401, {}), (200, {"token": "t2"}), (401, {})])
    runner = PipelineRunner(pb, ToolExecutor(http=http))
    result = await runner.run("p", _state())
    assert len(http.calls) == 3  # no second refresh
    assert not result.ok  # replayed 401 has no branch -> failed stop


async def test_failure_without_branch_stops() -> None:
    pb = _pb([PipelineStep(tool="hold", on={"ok": "j.done"})])
    runner = PipelineRunner(pb, ToolExecutor(http=FakeHttp([(503, {})])))
    result = await runner.run("p", _state())
    assert not result.ok and result.advance_to is None
    assert result.error_slot == {"error_context": "p:hold"}
