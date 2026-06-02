# SuperDialog — Embedding Guides

**Status:** Draft
**Parent:** [README.md](README.md)
**Purpose:** Show how SuperDialog drops into each common host environment.

---

## The shape of every embedding

In every host, three things stay the same:

1. **Construct a `DialogMachine`** with a flow, an LLM URI, and tools.
2. **Route inbound text** to `dialog_machine.turn(text)`.
3. **Send the reply text** back to the host's output channel.

That's it. The host varies; the SuperDialog code is identical.

---

## 1. CLI chatbot (testing / dev loop)

Zero infrastructure. Useful for prompt tuning, eval prep, and demos.

```python
import asyncio
from superdialog import create_dialog_flow, DialogMachine

async def main():
    flow = await create_dialog_flow(prompt="Confirm KYC.", llm="openai/gpt-5.1")
    dialog_machine = DialogMachine(
        flow=flow,
        llm="anthropic/claude-haiku-4-5",
        traversal_dir="./traversal_history",  # saves a JSON per session on completion
    )
    while True:
        user = input("> ")
        if user.strip() in ("quit", "exit"):
            break
        reply = await dialog_machine.turn(user)
        print(reply.text)

asyncio.run(main())
```

Pass `traversal_dir` to any `DialogMachine` — CLI, LiveKit, FastAPI, or any other host. A timestamped JSON file is written to that directory each time a session reaches a terminal node. Each file captures the full node path, every turn, and all collected slot values. Useful for inspecting flow behaviour, building eval datasets, and auditing production conversations.

Or use the bundled CLI:

```
superdialog chat kyc.json
```

**When to use:** during initial flow design, before any voice infrastructure is set up.

---

## 2. LiveKit

SuperDialog ships a `DialogMachineLLM` plugin that wires a `DialogMachine`
into a LiveKit `Agent` via the `llm=` parameter (the same shape LiveKit's
own `livekit-plugins-langchain` uses).

```python
from livekit.agents import Agent, AgentSession
from superdialog import DialogMachine, Flow
from superdialog.adapters.livekit import DialogMachineLLM

dialog_machine = DialogMachine(
    flow=Flow.load("kyc.json"),
    llm="anthropic/claude-opus-4-7",
)

async def entrypoint(ctx):
    agent = Agent(llm=DialogMachineLLM(dialog_machine))
    await AgentSession().start(agent=agent, room=ctx.room)
```

LiveKit's `AgentSession` drives the conversation; `DialogMachineLLM`
translates between LiveKit's `ChatContext` and SuperDialog's `turn()` API.

**When to use:** you're already on LiveKit for media routing and want SuperDialog to manage turn-by-turn logic.

---

## 3. PipeCat

PipeCat's `FrameProcessor` base class shifts between releases, so
SuperDialog ships a factory rather than a subclass: `make_processor(dm)`
synthesises a concrete `FrameProcessor` against whichever PipeCat is
installed.

```python
from superdialog import DialogMachine, Flow
from superdialog.adapters.pipecat import make_processor

dialog_machine = DialogMachine(flow=Flow.load("kyc.json"), llm="openai/gpt-5.1")
processor = make_processor(dialog_machine)

# Compose into a PipeCat pipeline
pipeline = Pipeline([
    stt_processor,
    processor,
    tts_processor,
])
```

**When to use:** PipeCat-based voice stack; SuperDialog replaces hand-written LLM logic between STT and TTS.

---

## 4. FastAPI (text chatbot / REST endpoint)

For single-user or stateless `/turn` endpoints, use a `DialogMachine` directly:

```python
from fastapi import FastAPI
from superdialog import DialogMachine, Flow

app = FastAPI()
dialog_machine = DialogMachine(flow=Flow.load("kyc.json"), llm="openai/gpt-5.1")

@app.post("/turn")
async def turn(payload: dict):
    return {"reply": (await dialog_machine.turn(payload["text"])).text}
```

For **multi-user** or **multi-worker** deployments, route per-conversation
state through a `SessionWorker` so any request can land on any worker and
resume the right conversation:

```python
from fastapi import FastAPI
from superdialog import DialogMachine, Flow, SessionWorker, InMemorySessionStore

app = FastAPI()
flow = Flow.load("kyc.json")
worker = SessionWorker(
    agent_factory=lambda: DialogMachine(flow=flow, llm="openai/gpt-5.1"),
    store=InMemorySessionStore(),    # swap for RedisSessionStore in production
)

@app.post("/turn")
async def turn(payload: dict):
    async with worker.acquire(payload["session_id"]) as h:
        result = await h.turn(payload["text"])
    return {"reply": result.text}
```

The `SessionWorker` multiplexes N concurrent sessions, each with its own
`DialogMachine`, sharing the immutable `Flow` by reference. Concurrent
requests for different `session_id`s run in parallel; concurrent requests
for the same id serialise via a per-session lock.

Mount on Intercom-style chat widget, WhatsApp webhook, SMS gateway, or anywhere HTTP fits.

**When to use:** non-voice deployments — text-only chatbot, support widget, async messaging.

---

## 5. Unpod Voice Infrastructure

This is the production voice path. SuperDialog runs on the developer's machine via a WebSocket runner; Unpod's infra connects to it.

```python
from superdialog import DialogMachine, Flow
from superdialog.adapters import WebSocketRunner

dialog_machine = DialogMachine(flow=Flow.load("kyc.json"), llm="anthropic/claude-opus-4-7")

WebSocketRunner(
    dialog_machine=dialog_machine,
    agent_id="kerali-kyc-bot",      # registers with Unpod
    api_key=os.environ["UNPOD_API_KEY"],
).serve(port=8080)
```

Then on Unpod side, the Identity binds the inbound number to this agent. When a call lands, Unpod connects to your WSS endpoint, streams text in, and sends agent text out for TTS. See [voice-infra/01-architecture.md](../voice-infra/01-architecture.md) for the full picture.

**When to use:** you want voice + numbers + speech infrastructure without writing telephony code.

---

## 6. Unit tests

`DialogMachine.turn` is async; tests use `pytest-asyncio` (or `anyio`).
The `state` property returns `{"node_id": ..., "slots": ...}` — read
collected data through `state["slots"]`.

```python
import pytest
from superdialog import DialogMachine, Flow

@pytest.mark.asyncio
async def test_kyc_flow_collects_aadhaar():
    machine = DialogMachine(
        flow=Flow.load("kyc.json"),
        llm="anthropic/claude-haiku-4-5",
    )
    reply = await machine.turn("मेरा आधार 1234 से शुरू होता है")
    assert "धन्यवाद" in reply.text or "thank" in reply.text.lower()
    assert machine.state["slots"].get("aadhaar_last_4") == "1234"
```

**When to use:** always. Because SuperDialog is text-only, every dialog is a unit-testable function. This is the killer feature vs voice-coupled frameworks where tests need audio fixtures.

---

## 7. Custom integration (anything else)

The interface is minimal: pass text in, get text out. **Note that
`turn(...)` is always async** — wrap it in an event loop for sync hosts.

```python
import asyncio

# IRC (sync handler)
def on_message(msg):
    reply = asyncio.run(dialog_machine.turn(msg.body))
    return reply.text

# Slack (sync handler)
@slack_app.message(...)
def handle(message, say):
    reply = asyncio.run(dialog_machine.turn(message["text"]))
    say(reply.text)

# Discord (async handler — preferred)
@discord_bot.event
async def on_message(message):
    reply = await dialog_machine.turn(message.content)
    await message.channel.send(reply.text)
```

For high-throughput hosts that hand you many concurrent conversations,
prefer a `SessionWorker` per process and route per-conversation state
through `worker.acquire(session_id)` — see §4 above.

> **Note on sync hosts:** wrapping every `dialog_machine.turn(...)` in
> `asyncio.run` creates a fresh event loop per call. For sustained traffic
> this is wasteful; either route through `SessionWorker` from an existing
> async runtime, or maintain a single long-lived loop. A dedicated
> `SyncDialogMachine` wrapper is planned for v0.3.

---

## Summary

| Host | Adapter needed | LoC |
|---|---|---|
| CLI | None — direct `input()`/`print()` loop or `superdialog chat` | ~5 |
| LiveKit | `on_user_message` override | ~8 |
| PipeCat | `DialogMachineProcessor` | ~12 |
| FastAPI | None — direct route | ~6 |
| Unpod Voice Infra | `WebSocketRunner` | ~6 |
| Unit test | None — direct calls | ~3 |
| Custom (Slack, Discord, IRC, etc.) | None — direct callback | ~3 |

The library does one thing well: text in, text out. Everything else is host code.
