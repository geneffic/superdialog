import anyio

from superdialog.playbook.events import AdvanceEvent, EventLog, UtteranceEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.state import ConversationState
from superdialog.playbook.talker import FILLER, HOLD_LINE, RECOVERY_LINE, Talker
from tests.playbook.test_models import MINIMAL_YAML


class StreamLLM:
    def __init__(self, chunks: list[str], fail_times: int = 0) -> None:
        self.chunks = chunks
        self.fail_times = fail_times
        self.calls = 0

    async def stream(self, messages, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("llm down")
        for c in self.chunks:
            yield c


def _state(checkpoint: str) -> tuple[Playbook, ConversationState]:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint=checkpoint, rule="init")
    )
    log.append(UtteranceEvent(role="user", text="hello"))
    return pb, ConversationState.fold(log, playbook=pb)


async def test_streams_tokens_and_reports_version() -> None:
    pb, state = _state("booking.collect")
    llm = StreamLLM(["Sure", ", which", " city?"])
    talker = Talker(pb, llm)
    chunks = [c async for c in talker.speak(state)]
    assert "".join(c.text for c in chunks) == "Sure, which city?"
    assert chunks[-1].spoke_from_version == state.version


async def test_say_verbatim_bypasses_llm() -> None:
    pb, state = _state("booking.confirm")
    llm = StreamLLM(["should not be called"])
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert "".join(c.text for c in chunks) == "Your booking is held."
    assert llm.calls == 0


async def test_failure_retries_once_then_recovers() -> None:
    pb, state = _state("booking.collect")
    flaky = Talker(pb, StreamLLM(["ok!"], fail_times=1))
    assert "".join([c.text async for c in flaky.speak(state)]) == "ok!"
    dead = Talker(pb, StreamLLM([], fail_times=99))
    assert RECOVERY_LINE in "".join([c.text async for c in dead.speak(state)])


async def test_hard_gate_filler_then_speech() -> None:
    pb, state = _state("booking.confirm")
    event = anyio.Event()

    async def wait_director() -> ConversationState:
        await event.wait()
        return state

    talker = Talker(pb, StreamLLM([]), barrier_timeout=0.05, hold_timeout=10.0)
    received: list[str] = []

    async def consume() -> None:
        async for c in talker.speak(state, director_done=wait_director):
            received.append(c.text)

    async with anyio.create_task_group() as tg:
        tg.start_soon(consume)
        await anyio.sleep(0.2)  # past barrier_timeout, director still pending
        assert any(FILLER in t for t in received)  # filler already emitted
        event.set()
    assert "".join(received).endswith("Your booking is held.")


async def test_hard_gate_hold_line_when_director_never_comes() -> None:
    pb, state = _state("booking.confirm")

    async def never() -> ConversationState:
        await anyio.sleep(60)
        return state

    talker = Talker(pb, StreamLLM([]), barrier_timeout=0.02, hold_timeout=0.05)
    received = [c.text async for c in talker.speak(state, director_done=never)]
    assert any(HOLD_LINE in t for t in received)
