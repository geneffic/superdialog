# SuperDialog — Architecture

**Status:** Draft
**Parent:** [README.md](README.md)
**Purpose:** Internal design of the framework. Components, contracts, data shapes.

---

## 1. Library shape

One Python package. No services, no daemons. Everything in-process.

```
superdialog/
  ├─ flow/                # Flow graph: nodes, edges, serialization
  ├─ machine/             # DialogStateMachine engine
  ├─ dialog_machine.py    # Public DialogMachine facade
  ├─ agent.py             # Agent Protocol + TurnResult
  ├─ agents/              # LLMAgent, LangChainAgent (non-DM brains)
  ├─ session/             # Session, SessionHandle, SessionWorker, stores, locks
  ├─ chat_context.py      # ChatContext, ChatMessage (LiveKit-aligned)
  ├─ flow_state.py        # FlowState (DM-specific runtime state)
  ├─ llm/                 # Model URI resolver and provider adapters
  ├─ tools/               # Python / HTTP / MCP tool wrappers
  ├─ cli/                 # `superdialog chat / flow lint / flow draw / flow generate`
  └─ adapters/            # LiveKit, PipeCat, FastAPI, WebSocket
```

> **Not in v0.2:** `eval/` (planned v0.3); distributed stores
> (`RedisSessionStore`, `FileSessionStore`, `SQLiteSessionStore`,
> `RedisLockBackend`) planned v0.3; true streaming inference planned v0.4.

## 2. Core components

### 2.1 Flow

A directed graph: `nodes` (states), `edges` (transitions), `metadata` (prompts, tool calls, branches).

```python
flow = await create_dialog_flow(
    prompt="Confirm KYC. Ask for Aadhaar last 4 digits.",
    llm="openai/gpt-5.1",
)
flow.save("kyc.json")            # JSON-serializable, version-controllable
flow = Flow.load("kyc.json")
```

`create_dialog_flow` is async. From a sync entry point, wrap it in
`asyncio.run(...)`.

Authoring options:
- **Prompt-only.** `create_dialog_flow(prompt=..., llm=...)` — LLM bootstraps a graph.
- **Hand-built.** Construct nodes and edges directly.
- **Multi-flow.** A `FlowSet` is a collection of small flows; runtime switches between them.

The LLM in `create_dialog_flow` is used **at construction time only**, never at runtime.

### 2.2 DialogMachine

The runtime. Owns conversation memory, model URI, tools.

```python
dialog_machine = DialogMachine(
    flow=flow,                # or FlowSet=...
    llm="anthropic/claude-opus-4-7",
    tools=[...],
)
```

Contract (single async method with streaming flag):

```python
async def turn(
    text: str,
    context: dict | None = None,
    stream: bool | Literal["text"] = False,
) -> Turn | AsyncIterator[StreamChunk]
```

- `stream=False` (default): `await dm.turn(text)` → complete `Turn` (`text`, `tool_calls`, `metadata`)
- `stream=True` / `stream="text"`: `stream = await dm.turn(text, stream=True)` → async iterator of `StreamChunk`

Additional methods:
- `reset()` — clear memory, restart from initial state
- `set_llm(uri)` — swap model (applies to next turn)
- `switch_flow(name)` — hot-swap to a named flow within the FlowSet
- `assist(text)` — push a system message that takes effect next turn (was `inject_system`; old name kept as deprecated alias)

### 2.3 Model URI resolver

LiveKit/litellm-style URIs.

| URI | Routes to |
|---|---|
| `openai/gpt-5.1` | OpenAI |
| `anthropic/claude-opus-4-7` | Anthropic |
| `anthropic/claude-haiku-4-5` | Anthropic |
| `google/gemini-2.5-pro` | Google |
| `groq/llama-3.3-70b` | Groq |
| `bedrock/<model>` | AWS Bedrock |
| `vllm/<model>@<host>` | Self-hosted vLLM |
| `ollama/<model>@<host>` | Self-hosted Ollama |
| `openrouter/<vendor>/<model>` | OpenRouter |
| `unpod/<vertical>` | Unpod-hosted vertical LLMs (e.g. `unpod/insurance-v1`) — **planned, not implemented in v0.2** |
| `custom/<name>` | Developer-registered via `register_llm_provider(...)` |

Custom providers (process-global registry):

```python
register_llm_provider(
    name="kerali-internal",
    base_url="https://llm.kerali.io/v1",
    api_key=os.environ["KERALI_LLM_KEY"],
    api_style="openai",
)
# Usable as "custom/kerali-internal/llama-3-70b-tuned"
```

### 2.4 Tools

Three shapes, one interface:

```python
PythonTool.of(my_local_function)                       # infer id/name/schema
HttpTool(
    id="lookup", name="lookup", description="...",
    url="https://api.kerali.io/lookup",
)
MCPTool(id="...", name="...", description="...", server="https://mcp.kerali.io")
```

Tools are passed to `DialogMachine(tools=[...])`. Execution is in-process
for `PythonTool`, network for `HttpTool` and `MCPTool`. Tool results
merge into `node_slots` / `userdata` and (optionally) trigger an edge
transition when the handler returns
`ToolResult(transition_edge_id="...")`.

### 2.5 Eval harness

> **Status: planned for v0.3.** The interface below is the design target.

The differentiating feature. Runs a corpus of (utterance, expected outcome) pairs against any model URI and any flow.

```python
# v0.3 (planned)
from superdialog.eval import Eval

eval = Eval(
    flow=flow,
    corpus="tests/kyc-corpus.jsonl",
    llms=["openai/gpt-5.1", "anthropic/claude-haiku-4-5", "groq/llama-3.3-70b"],
    metrics=["accuracy", "latency_p95", "tool_call_correctness"],
)
report = eval.run()
report.save("reports/kyc-2026-05-19.md")
```

Output: per-LLM comparison, latency distributions, failure modes by category. Used by developers to choose models and by Unpod (when bundled) to suggest defaults.

### 2.6 CLI

```
superdialog chat <flow.json>           # Interactive terminal chat (shipped)
superdialog flow lint <flow.json>      # Validate graph structure (shipped)
superdialog flow draw <flow.json>      # Render Mermaid diagram (shipped)
superdialog flow generate "<prompt>"   # Bootstrap flow.json (shipped)
superdialog eval <flow.json> <corpus>  # Run eval harness (planned v0.3)
```

The `chat` subcommand is critical: it lets a developer test a dialog machine **the same way an end user would experience it on a phone**, but without any voice infrastructure. No setup, no API keys for Unpod, no phone number.

## 3. Adapter pattern

Adapters live in `superdialog.adapters` and are thin shims.

### 3.1 LiveKit

`DialogMachineLLM` is a LiveKit `Agent.llm=` plugin — same pattern as
`livekit-plugins-langchain`. The LiveKit `AgentSession` drives the
conversation; the plugin translates between LiveKit's `ChatContext` and
SuperDialog's `turn()` API.

```python
from livekit.agents import Agent, AgentSession
from superdialog import DialogMachine
from superdialog.adapters.livekit import DialogMachineLLM

async def entrypoint(ctx):
    dm = DialogMachine(flow=flow, llm="openai/gpt-5.1")
    agent = Agent(llm=DialogMachineLLM(dm))
    await AgentSession().start(agent=agent, room=ctx.room)
```

### 3.2 PipeCat

PipeCat's `FrameProcessor` API shifts between releases, so SuperDialog
exposes a factory rather than a subclass: `make_processor(dm)`
synthesises the right `FrameProcessor` against whichever PipeCat is
installed.

```python
from superdialog import DialogMachine
from superdialog.adapters.pipecat import make_processor

dm = DialogMachine(flow=flow, llm="openai/gpt-5.1")
processor = make_processor(dm)
# add `processor` to your PipeCat pipeline
```

### 3.3 FastAPI

```python
from fastapi import FastAPI
from superdialog import DialogMachine

app = FastAPI()
dialog_machine = DialogMachine(flow=flow, llm="openai/gpt-5.1")

@app.post("/turn")
async def turn(payload: dict):
    reply = await dialog_machine.turn(payload["text"])
    return {"reply": reply.text}
```

For multi-worker / multi-user deployments, route per-conversation state
through a `SessionWorker` — see §4 *Memory and Sessions*.

### 3.4 WebSocket runner

For integration with Unpod Voice Infra (or any host that prefers a network endpoint over a Python import):

```python
from superdialog.adapters import WebSocketRunner

WebSocketRunner(
    dialog_machine=dialog_machine,
    agent_id="kerali-kyc-bot",
).serve(port=8080)
```

This adapter is the bridge between SuperDialog and Voice Infra. Voice Infra connects to this WSS endpoint; everything before that point is pure SuperDialog.

## 4. Memory and Sessions

A live `DialogMachine` keeps its conversation in memory for the lifetime of
the instance. For workloads where the conversation outlives the process
(async HTTP handlers, multi-worker FastAPI, day-long chat), wrap the DM in
a `SessionWorker`:

```python
from superdialog import DialogMachine, SessionWorker, InMemorySessionStore

worker = SessionWorker(
    agent_factory=lambda: DialogMachine(flow=flow, llm="openai/gpt-5.1"),
    store=InMemorySessionStore(),
)

async with worker.acquire(session_id) as h:
    reply = await h.turn(text)
```

The Worker constructs one `DialogMachine` per active session, shares the
immutable `Flow` by reference, multiplexes N concurrent sessions, and
persists each session's `ChatContext` + `FlowState` to a pluggable
`SessionStore`.

**v0.2 store backends:** `InMemorySessionStore` (process-local),
`NullSessionStore` (no persistence — use for voice / ephemeral).
**v0.3 (planned):** `RedisSessionStore`, `FileSessionStore`,
`SQLiteSessionStore`, `RedisLockBackend` for distributed deployments.

For voice scenarios (one DialogMachine per call), `NullSessionStore` keeps
the existing single-call lifecycle. For long-lived chat, persistent
backends land in v0.3.

## 5. Streaming

`await dialog_machine.turn(text, stream=True)` returns an async iterator
of `StreamChunk`. The coroutine must be awaited out before iterating:

```python
stream = await dialog_machine.turn("hello", stream=True)
async for chunk in stream:
    print(chunk.text, end="")
```

Streaming is opt-in per call. The non-streaming `await dialog_machine.turn(text)`
form returns a complete `Turn` and is the default for unit tests and CLI usage.

> **v0.2 implementation note:** the response is resolved in one shot, then
> emitted as whitespace-delimited chunks. True provider-level streaming
> inference (first-token latency) is planned for v0.4.

## 6. What lives outside this library

- Audio processing
- STT, TTS
- Telephony, SIP, RTP
- Media servers, Rooms (in the WebRTC sense)
- Numbers, voice profiles
- Billing

All of those are Voice Infra's problem. SuperDialog ends at text in, text out.
