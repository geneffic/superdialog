from superdialog.playbook.events import (
    EventLog,
    SlotWriteEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from superdialog.playbook.models import SlotSpec, ToolSpec
from superdialog.playbook.state import ConversationState
from superdialog.playbook.toolexec import ToolExecutor, coerce_args


class FakeHttp:
    def __init__(self, responses: list[tuple[int, dict]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: dict,
        body: dict | None,
        timeout: float,
    ) -> tuple[int, dict]:
        self.calls.append(
            {"method": method, "url": url, "headers": headers, "body": body}
        )
        return self.responses.pop(0)


def _state(**slots) -> ConversationState:
    log = EventLog()
    for k, v in slots.items():
        log.append(SlotWriteEvent(key=k, value=v, status="confirmed", by="director"))
    state = ConversationState.fold(log)
    state.env["API_BASE_URL"] = "https://api.test"
    state.env["ACCESS_TOKEN"] = "tok-1"
    return state


HOLD = ToolSpec(
    id="hold_slot",
    method="POST",
    url="{{ env.API_BASE_URL }}/slots/hold",
    headers={"Authorization": "Bearer {{ env.ACCESS_TOKEN }}"},
    body={"slot_id": "{{ slots.slot_id }}", "players": "{{ slots.players }}"},
    store_response_as="hold_result",
    env_updates={"hold_id": "data.hold_id"},
)


async def test_executes_and_stores_result_and_env() -> None:
    http = FakeHttp([(200, {"data": {"hold_id": "h-77"}})])
    ex = ToolExecutor(http=http)
    events = await ex.execute(HOLD, _state(slot_id="s1", players=4))
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent", "EnvWriteEvent"]
    assert http.calls[0]["url"] == "https://api.test/slots/hold"
    assert http.calls[0]["headers"]["Authorization"] == "Bearer tok-1"
    result = events[1]
    assert (
        isinstance(result, ToolResultEvent)
        and result.ok
        and result.store_as == "hold_result"
    )
    assert events[2].key == "hold_id" and events[2].value == "h-77"


async def test_when_predicate_skips() -> None:
    spec = HOLD.model_copy(update={"when": "slots.player_id"})
    ex = ToolExecutor(http=FakeHttp([]))
    events = await ex.execute(spec, _state(slot_id="s1"))  # no player_id
    assert events == []


async def test_run_once_skips_second_execution() -> None:
    spec = HOLD.model_copy(update={"run_once": True})
    state = _state(slot_id="s1", players=2)
    state.tool_call_counts["hold_slot"] = 1  # already ran once
    ex = ToolExecutor(http=FakeHttp([]))
    assert await ex.execute(spec, state) == []


async def test_http_error_yields_failed_result() -> None:
    ex = ToolExecutor(http=FakeHttp([(503, {"error": "upstream"})]))
    events = await ex.execute(HOLD, _state(slot_id="s1", players=2))
    result = events[1]
    assert result.ok is False and result.status == 503


async def test_template_error_yields_failed_result() -> None:
    # Broken template syntax: degrade to a failed result, never raise.
    bad = HOLD.model_copy(update={"url": "{{ env.API_BASE_URL "})
    http = FakeHttp([])
    ex = ToolExecutor(http=http)
    events = await ex.execute(bad, _state(slot_id="s1", players=2))
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]
    result = events[1]
    assert isinstance(result, ToolResultEvent) and result.ok is False
    assert http.calls == []  # bad template never reaches HTTP
    assert "template error" in (result.error or "")

    # SSTI payload: the sandbox blocks the attribute walk; the payload is
    # NOT executed (a plain Environment would render a function repr with
    # a memory address like "at 0x...").
    ssti = HOLD.model_copy(update={"url": "{{ cycler.__init__ }}/x"})
    http2 = FakeHttp([(200, {})])
    events2 = await ToolExecutor(http=http2).execute(
        ssti, _state(slot_id="s1", players=2)
    )
    assert http2.calls[0]["url"] == "/x"  # unsafe attr rendered empty
    assert "0x" not in http2.calls[0]["url"]
    assert [type(e).__name__ for e in events2] == [
        "ToolCallEvent",
        "ToolResultEvent",
    ]


async def test_python_tool_and_arg_coercion() -> None:
    spec = ToolSpec(
        id="score",
        type="python",
        store_response_as="score_result",
        args={"n": SlotSpec(type="int")},
    )
    seen: list[object] = []

    async def score_fn(args: dict, state: ConversationState) -> dict:
        seen.append(args["n"])
        assert isinstance(args["n"], int) and args["n"] == 7
        return {"score": args["n"] * 2}

    ex = ToolExecutor(http=FakeHttp([]), python_tools={"score": score_fn})
    events = await ex.execute(spec, _state(), args={"n": "7"})
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]
    result = events[1]
    assert isinstance(result, ToolResultEvent) and result.ok
    assert result.data == {"score": 14}
    assert seen == [7]

    async def boom_fn(args: dict, state: ConversationState) -> dict:
        raise RuntimeError("python tool exploded")

    spec_boom = spec.model_copy(update={"id": "boom"})
    ex2 = ToolExecutor(http=FakeHttp([]), python_tools={"boom": boom_fn})
    events2 = await ex2.execute(spec_boom, _state())
    result2 = events2[1]
    assert isinstance(result2, ToolResultEvent) and result2.ok is False
    assert "exploded" in (result2.error or "")


async def test_unregistered_python_tool_degrades() -> None:
    spec = ToolSpec(id="ghost", type="python", store_response_as="ghost_result")
    ex = ToolExecutor(http=FakeHttp([]))  # no python tools registered
    events = await ex.execute(spec, _state())
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]
    result = events[1]
    assert isinstance(result, ToolResultEvent) and result.ok is False
    assert "not registered" in (result.error or "")
    assert result.store_as == "ghost_result"


async def test_error_strings_carry_type() -> None:
    async def slow_fn(args: dict, state: ConversationState) -> dict:
        raise TimeoutError()  # str(exc) would be ""

    spec = ToolSpec(id="slow", type="python", store_response_as="slow_result")
    ex = ToolExecutor(http=FakeHttp([]), python_tools={"slow": slow_fn})
    events = await ex.execute(spec, _state())
    result = events[1]
    assert isinstance(result, ToolResultEvent) and result.ok is False
    assert (result.error or "").startswith("TimeoutError")


async def test_coercion_failure_shape() -> None:
    spec = ToolSpec(
        id="score",
        type="python",
        store_response_as="score_result",
        args={"n": SlotSpec(type="int")},
    )
    ex = ToolExecutor(http=FakeHttp([]))
    events = await ex.execute(spec, _state(), args={"n": "7.5"})
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]
    result = events[1]
    assert isinstance(result, ToolResultEvent)
    assert result.ok is False and result.status is None
    assert (result.error or "").startswith("bad args")


async def test_env_update_missing_path_skipped() -> None:
    # env_updates wants data.hold_id, but the response has no hold_id.
    http = FakeHttp([(200, {"data": {}})])
    ex = ToolExecutor(http=http)
    events = await ex.execute(HOLD, _state(slot_id="s1", players=2))
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]  # no EnvWriteEvent


async def test_secret_body_keys_redacted() -> None:
    spec = ToolSpec(
        id="auth",
        method="POST",
        url="{{ env.API_BASE_URL }}/auth",
        body={"client_secret": "{{ env.CS }}", "city": "x"},
        store_response_as="auth_result",
    )
    state = _state()
    state.env["CS"] = "s3cr3t"
    http = FakeHttp([(200, {})])
    ex = ToolExecutor(http=http)
    events = await ex.execute(spec, state)
    call = events[0]
    assert isinstance(call, ToolCallEvent)
    assert call.args["body"]["client_secret"] == "***"  # event log masked
    assert call.args["body"]["city"] == "x"  # non-secret keys intact
    assert http.calls[0]["body"]["client_secret"] == "s3cr3t"  # real body sent


async def test_redaction_recursive_and_broad() -> None:
    spec = ToolSpec(
        id="nested",
        method="POST",
        url="{{ env.API_BASE_URL }}/nested",
        body={"auth": {"client_secret": "s"}, "items": [{"jwt": "x", "city": "Pune"}]},
        store_response_as="nested_result",
    )
    http = FakeHttp([(200, {})])
    ex = ToolExecutor(http=http)
    events = await ex.execute(spec, _state())
    call = events[0]
    assert isinstance(call, ToolCallEvent)
    assert call.args["body"]["auth"] == "***"  # broad denylist: auth masked
    assert call.args["body"]["items"][0]["jwt"] == "***"  # recursed into list
    assert call.args["body"]["items"][0]["city"] == "Pune"  # non-secret intact
    assert http.calls[0]["body"] == {  # real body sent unmasked
        "auth": {"client_secret": "s"},
        "items": [{"jwt": "x", "city": "Pune"}],
    }


async def test_url_redaction() -> None:
    spec = ToolSpec(
        id="geo",
        method="GET",
        url="https://u:p@api.test/x?api_key={{ env.K }}&city=pune",
        store_response_as="geo_result",
    )
    state = _state()
    state.env["K"] = "sek"
    http = FakeHttp([(200, {})])
    ex = ToolExecutor(http=http)
    events = await ex.execute(spec, state)
    call = events[0]
    assert isinstance(call, ToolCallEvent)
    recorded = call.args["url"]
    assert "u:p@" not in recorded  # userinfo stripped
    assert "sek" not in recorded
    assert "api_key=***" in recorded  # secret param masked, key kept
    assert "city=pune" in recorded  # non-secret param intact
    # the REAL url (userinfo + secrets) still went to http
    assert http.calls[0]["url"] == "https://u:p@api.test/x?api_key=sek&city=pune"


def test_bool_coercion_false() -> None:
    assert coerce_args({"f": "false"}, {"f": SlotSpec(type="bool")})["f"] is False
