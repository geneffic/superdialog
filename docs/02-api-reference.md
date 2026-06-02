
# SuperDialog — API Reference

**Status:** Draft
**Parent:** [README.md](README.md)
**Audience:** Developers writing code against the library.

---

## Construction

### `async create_dialog_flow(prompt, llm, **kwargs) -> Flow`

Bootstrap a flow graph from a prompt using a one-shot LLM call.

```python
import asyncio
from superdialog import create_dialog_flow

flow = await create_dialog_flow(            # inside async code
    prompt="Confirm appointment. Ask if Friday 4pm works; offer 5pm if not.",
    llm="openai/gpt-5.1",
)
# or, from a sync entry point:
# flow = asyncio.run(create_dialog_flow(prompt=..., llm=...))
```

The `llm` parameter is used **only at construction**. The runtime model is set on `DialogMachine`.

### `Flow.save(path)` / `Flow.load(path)`

Serialize / deserialize. JSON. Version-controllable.

### `FlowSet(flows: dict[str, Flow])`

Container for multiple small flows. Switch between them at runtime.

```python
flowset = FlowSet({
    "main": main_flow,
    "escalation": escalation_flow,
    "billing": billing_flow,
})
```

---

## DialogMachine

### Construction

```python
DialogMachine(
    flow: Flow | FlowSet,
    llm: str,                           # model URI
    tools: list[Tool] | None = None,
    memory: Memory | None = None,       # default: in-memory
    config: dict | None = None,         # max_tokens, temperature, etc.
    traversal_dir: str | Path | None = None,  # auto-save traversal JSON here on session end
)
```

Set `traversal_dir` to a directory path and the machine will write a timestamped JSON file recording every node visited, every turn, and slot values collected — automatically when `is_complete` becomes `True`. Useful for debugging flows, building eval datasets, and auditing production conversations.

### `async turn(text, context=None, stream=False) -> Turn | AsyncIterator[StreamChunk]`

The primary method. One method, one parameter for streaming mode. **Always
async** — there is no synchronous wrapper. Drive it from `asyncio.run(...)`
or any async runtime.

```python
# Non-streaming
reply = await dialog_machine.turn("hello")
print(reply.text)

# Streaming — `turn(stream=True)` returns a coroutine that resolves to an
# async iterator, so the iterator must be awaited out of the coroutine first.
stream = await dialog_machine.turn("hello", stream=True)
async for chunk in stream:
    print(chunk.text, end="")
```

`Turn` has:
- `text: str`
- `tool_calls: list[ToolCall]`
- `metadata: dict` (latency, tokens, model used)

> **Streaming policy (v0.2):** the v0.2 implementation resolves the turn
> in one shot, then surfaces the response as whitespace-delimited chunks.
> True provider-level streaming inference is planned for v0.4. The chunk
> shape (`StreamChunk(text, done, turn)`) is stable.

### `reset()`

Clear conversation memory, restart from the flow's initial node. Useful between independent conversations on the same `DialogMachine` instance.

### `set_llm(uri: str)`

Hot-swap the model. Applies to next turn (in-flight streaming continues on the old model).

```python
dialog_machine.set_llm("anthropic/claude-haiku-4-5")
```

### `switch_flow(name: str)`

If the machine was constructed with a `FlowSet`, switch to a named flow. State is reset by default; pass `preserve_memory=True` to keep history.

```python
dialog_machine.switch_flow("escalation")
```

### `assist(text: str)`

Push a system-level instruction that takes effect next turn. Used for mid-call context injection.

```python
dialog_machine.assist("Customer is upset. Be especially empathetic.")
```

> `inject_system(...)` is preserved as a deprecated alias and emits a
> `DeprecationWarning` on call. Slated for removal in v0.4.

---

## Sessions (v0.2)

Sessions add a lifecycle and persistence layer on top of `DialogMachine` (and other
`Agent`-protocol-compatible brains). Use them when you need to **resume a
conversation across process boundaries** (async HTTP handlers, multi-worker
deployments, long-lived chat across days).

### The Agent Protocol

```python
class Agent(Protocol):
    async def turn(text: str, *, stream: bool = False) -> TurnResult | AsyncIterator[StreamChunk]
    def assist(text: str) -> None
    @property def chat_ctx -> ChatContext
    def load_chat_ctx(ctx: ChatContext) -> None
```

`DialogMachine`, `LLMAgent`, and `LangChainAgent` all implement this Protocol.

### `Session` and `SessionWorker`

```python
from superdialog import DialogMachine, SessionWorker, InMemorySessionStore

flow = Flow.load("kyc.json")
tools = [PythonTool.of(lookup_customer)]

# One Worker per process; one Agent (and one Session) per active conversation.
worker = SessionWorker(
    agent_factory=lambda: DialogMachine(flow=flow, llm="openai/gpt-5.1", tools=tools),
    store=InMemorySessionStore(),
)

async with worker.acquire("user-42") as h:
    result = await h.turn("hello")
    h.assist("Customer sounds upset; be empathetic.")
```

- **`SessionWorker(agent_factory, store, lock_backend, max_sessions=1000)`** —
  process-level multiplexer. Calls `agent_factory()` once per new session.
- **`worker.acquire(session_id)`** — async context manager. Loads or creates
  the session, locks it for the duration of the block, persists state on exit.
- **`SessionHandle`** — yielded inside the with-block. `.turn(text, *, stream)`,
  `.assist(text)`, `.state`.
- **`Session`** — the durable data (`id`, `chat_ctx`, `flow_state`,
  `metadata`). Not normally constructed directly.

### `ChatContext` and `FlowState`

`ChatContext` is LiveKit-aligned message history:

```python
@dataclass class ChatMessage: role: Literal["system","user","assistant","tool"]; content: str
@dataclass class ChatContext: items: list[ChatMessage]
```

`FlowState` is DM-specific runtime state (current node, slots, etc.) — used
only when the session's brain is a `DialogMachine`. Sessions bound to
non-DM brains have `flow_state=None`.

### `SessionStore` and `LockBackend`

Pluggable backends:

| Protocol | Ships in v0.2 | Planned |
|---|---|---|
| `SessionStore` | `InMemorySessionStore`, `NullSessionStore` | `RedisSessionStore`, `FileSessionStore`, `SQLiteSessionStore` |
| `LockBackend` | `AsyncioLockBackend` | `RedisLockBackend` |

`InMemorySessionStore` persists for the process lifetime; `NullSessionStore`
drops every write — use it for voice (one DM per call) where persistence
is unwanted.

### Alternative agents (non-DM)

```python
from superdialog import LLMAgent, SessionWorker

worker = SessionWorker(
    agent_factory=lambda: LLMAgent(llm="openai/gpt-5.1", system_prompt="Be helpful."),
    store=InMemorySessionStore(),
)
```

`LLMAgent` is a raw chat brain — no flow, no slots. Useful when you want
sessions/persistence/concurrency but no state-machine opinion.

`LangChainAgent` (requires `pip install superdialog[langchain]`) wraps an
async LangChain runnable.

### `assist(text)` — pushing system messages

Both `DialogMachine.assist(...)` and `SessionHandle.assist(...)` are the
canonical way to push a system-level instruction mid-conversation.
`DialogMachine.inject_system` remains as a deprecated alias (slated for
removal in v0.4) and emits a `DeprecationWarning` on call.

---

## Tools

### `PythonTool.of(fn, name=None, description=None)`

The convenience constructor. Infers `id`, `name`, `description`, and
`input_schema` from the function's signature and docstring.

```python
def lookup_customer(customer_id: str) -> dict:
    """Look up customer record by ID."""
    return crm.get(customer_id)

tool = PythonTool.of(lookup_customer)
```

The bare `PythonTool(id=..., name=..., description=..., fn=...)` constructor
is also available when you need to override identity or schema explicitly.

### `HttpTool(id, name, description, url, method="POST", auth=None, input_schema=None)`

```python
tool = HttpTool(
    id="lookup",
    name="lookup",
    description="Look up a customer by partial Aadhaar",
    url="https://api.kerali.io/customer/lookup",
    auth={"type": "bearer", "token": os.environ["KERALI_KEY"]},
)
```

`auth` accepts a dict in v0.2:
- `{"type": "bearer", "token": "..."}` — Bearer token in `Authorization`.

Additional auth shapes (`basic`, `api_key`, callable) are planned for v0.3.

### `MCPTool(id, name, description, server, input_schema=None)`

```python
tool = MCPTool(id="search", name="search", description="...", server="https://mcp.kerali.io")
```

> **Status (v0.2):** the MCPTool wrapper forwards `execute(args)` to
> `session.call_tool(self.id, args)` against the configured server.
> Auto-discovery and namespacing of *all* tools on an MCP server (so one
> `MCPTool` registration exposes every tool the server publishes) is
> planned for a follow-up.

---

## LLM provider registration

### `register_llm_provider(name, base_url, api_key, api_style="openai")`

Process-global. Once registered, the URI `custom/<name>/<model>` works in `set_llm()`, `DialogMachine(llm=...)`, and `create_dialog_flow(llm=...)`.

```python
register_llm_provider(
    name="kerali-internal",
    base_url="https://llm.kerali.io/v1",
    api_key=os.environ["KERALI_KEY"],
    api_style="openai",
)
dialog_machine = DialogMachine(flow=flow, llm="custom/kerali-internal/llama-3-70b-tuned")
```

---

## Eval

> **Status: planned for v0.3.** The `Eval` class is not shipped in v0.2.
> The interface below is the target surface; track the
> `superdialog-eval` change in `openspec/changes/` for progress.

```python
# v0.3 (planned)
eval = Eval(
    flow=flow,
    corpus="tests/kyc-corpus.jsonl",
    llms=["openai/gpt-5.1", "anthropic/claude-haiku-4-5"],
    metrics=["accuracy", "latency_p95"],
)
report = eval.run()
print(report.summary())
report.save("reports/2026-05-19.md")
```

Corpus will be JSONL with `{utterance, expected_response | expected_intent | expected_tool_call}` records; custom metrics will be passable as callables.

---

## Adapters

The actual module paths shipped in v0.2:

| Import | Purpose |
|---|---|
| `superdialog.adapters.livekit.DialogMachineLLM` | LiveKit `Agent(llm=...)` plugin (livekit-plugins-langchain-style) |
| `superdialog.adapters.pipecat.make_processor` | Factory that builds a PipeCat `FrameProcessor` |
| `superdialog.adapters.fastapi.FastAPIRouter` | Mountable FastAPI router exposing `/turn`, `/stream`, `/reset` |
| `superdialog.adapters.websocket.WebSocketRunner` | Standalone WSS server (Unpod Voice Infra) |

See `docs/03-embedding-guides.md` for working snippets per host.

---

## CLI

| Command | Purpose | Status |
|---|---|---|
| `superdialog chat <flow.json>` | Interactive REPL chat | shipped |
| `superdialog flow lint <flow.json>` | Validate graph | shipped |
| `superdialog flow draw <flow.json>` | Render Mermaid diagram | shipped |
| `superdialog flow generate "<prompt>" --llm openai/gpt-5.1` | Bootstrap flow.json from a prompt | shipped |
| `superdialog eval <flow.json> <corpus.jsonl>` | Run eval harness | planned (v0.3) |

---

## Worked example — end to end

A KYC bot built once, deployed four ways. Same `DialogMachine` object passes through every host.

```python
import asyncio
from superdialog import create_dialog_flow, DialogMachine, PythonTool

# ── 1. Bootstrap a flow from a prompt (one-shot LLM call at construction) ─
async def build_flow():
    flow = await create_dialog_flow(
        prompt="Verify customer KYC. Ask for Aadhaar last 4. Confirm DOB.",
        llm="openai/gpt-5.1",
    )
    flow.save("kyc.json")                     # version-control it
    return flow

flow = asyncio.run(build_flow())

# ── 2. Register a tool (Python callable; HTTP or MCP equally valid) ──────
def lookup_customer(aadhaar_last_4: str) -> dict:
    """Lookup customer by partial Aadhaar."""
    return crm.lookup_by_partial_aadhaar(aadhaar_last_4)

# ── 3. Build the runtime machine ─────────────────────────────────────────
dialog_machine = DialogMachine(
    flow=flow,
    llm="anthropic/claude-haiku-4-5",         # runtime model, cost lever
    tools=[PythonTool.of(lookup_customer)],
)

# ── 4a. Test as a CLI chatbot — no infrastructure needed ─────────────────
async def repl():
    while True:
        user = input("> ")
        if user.strip() in {"quit", "exit"}: break
        reply = await dialog_machine.turn(user)
        print(reply.text)

asyncio.run(repl())

# ── 4b. Or use the bundled CLI ───────────────────────────────────────────
#       $ superdialog chat kyc.json

# ── 5. Drop into LiveKit (LLM-plugin pattern; same dialog_machine) ───────
from livekit.agents import Agent, AgentSession
from superdialog.adapters.livekit import DialogMachineLLM

async def lk_entrypoint(ctx):
    agent = Agent(llm=DialogMachineLLM(dialog_machine))
    await AgentSession().start(agent=agent, room=ctx.room)

# ── 6. Or drop into PipeCat ──────────────────────────────────────────────
from superdialog.adapters.pipecat import make_processor
pipecat_node = make_processor(dialog_machine)

# ── 7. Or expose to Unpod Voice Infra via WSS runner ─────────────────────
from superdialog.adapters.websocket import WebSocketRunner
WebSocketRunner(
    dialog_machine=dialog_machine,
    agent_id="kerali-kyc-bot",                # bind this name in Unpod portal
).serve(port=8080)
```

| Step | Host | LoC added |
|---|---|---|
| 1-3 | (none — just construct the machine) | ~10 |
| 4 | CLI chatbot | ~3 |
| 5 | LiveKit agent | ~6 |
| 6 | PipeCat pipeline | ~2 |
| 7 | Unpod Voice Infra (WSS runner) | ~5 |

One `DialogMachine` instance, four hosts, one product surface.

For the **full Unpod Voice Infra journey** — portal config (voice profile, number, agent binding) alongside the SDK code — see [../voice-infra/journey-quickstart.md](../voice-infra/journey-quickstart.md).
