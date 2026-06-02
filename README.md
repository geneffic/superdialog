# SuperDialog

**Standalone dialog state machine framework. Pure text in, pure text out.**

SuperDialog is the **brain** layer for conversational systems. It turns a prompt
or a flow graph into a running dialog state machine — managing turn-by-turn
logic, tool calls, flow transitions, and conversation memory.

```
User text → DialogMachine.turn() → Agent reply text
```

Audio, STT, TTS, telephony, and media servers are out of scope — those belong to
voice infrastructure like LiveKit, PipeCat, or the Unpod Voice Platform.
SuperDialog ends at text in, text out.

> SuperDialog is to **conversation flow** what n8n is to **integration
> workflow** — a small, composable, eval-able runtime for orchestrating
> turn-by-turn logic. Where LangChain and LangGraph expose general agent
> primitives, SuperDialog focuses narrowly on the conversational state machine:
> who speaks next, what flow to switch to, when to call a tool, when to escalate.

---

## Why standalone

**The brain has natural reuse beyond voice.** A dialog state machine that runs a
customer-onboarding flow works the same whether the user is on a phone, a
WhatsApp thread, an Intercom widget, or a CLI test harness. Coupling it to
telephony forecloses every non-voice use case.

**The dependency direction matters.** Voice infrastructure should depend on
SuperDialog (as one brain option), not the other way around — keeping the
framework portable and the platform composable.

Because the interface is text-only, **every dialog is a unit-testable function.**
No audio fixtures, no API keys, no phone number to test a flow.

## Install

```bash
pip install superdialog
```

Install only the extras you need:

```bash
pip install superdialog[livekit]    # LiveKit adapter
pip install superdialog[pipecat]    # PipeCat adapter
pip install superdialog[fastapi]    # FastAPI adapter + uvicorn
pip install superdialog[ws]         # WebSocket runner
pip install superdialog[mcp]        # MCP tool support
pip install superdialog[langchain]  # LangChainAgent
```

## Quickstart

```python
import asyncio
from superdialog import create_dialog_flow, DialogMachine, Flow

# 1. Bootstrap a flow from a prompt (one-shot LLM call at construction).
#    The build LLM is used ONLY here — never at runtime.
async def build():
    flow = await create_dialog_flow(
        prompt="Confirm appointment. Ask if Friday 4pm works; offer 5pm if not.",
        llm="openai/gpt-5.1",
    )
    flow.save("appointment.json")        # JSON, version-controllable

asyncio.run(build())

# 2. Build the runtime machine (runtime model can differ from the build model).
dialog_machine = DialogMachine(
    flow=Flow.load("appointment.json"),
    llm="anthropic/claude-haiku-4-5",
)

# 3. Run a conversation.
async def chat():
    reply = await dialog_machine.turn("Hi, I'm calling about my appointment.")
    print(reply.text)

asyncio.run(chat())
```

Or skip the Python and use the bundled CLI:

```bash
superdialog chat appointment.json
```

## Add a tool

Three shapes, one interface — Python callables, HTTP endpoints, MCP servers:

```python
from superdialog import DialogMachine, Flow, PythonTool

def lookup_customer(phone_number: str) -> dict:
    """Look up customer record by phone number."""
    return crm.get_by_phone(phone_number)

dialog_machine = DialogMachine(
    flow=Flow.load("appointment.json"),
    llm="anthropic/claude-haiku-4-5",
    tools=[PythonTool.of(lookup_customer)],   # schema inferred from signature
)
```

`HttpTool(...)` and `MCPTool(...)` plug in the same way. Tool results merge into
the flow's slots and can trigger an edge transition.

## Model URIs

Pick a provider per machine with a LiveKit/litellm-style URI — and swap it at
runtime with `dialog_machine.set_llm(uri)`:

| URI | Routes to |
|---|---|
| `openai/gpt-5.1` | OpenAI |
| `anthropic/claude-opus-4-7` | Anthropic |
| `google/gemini-2.5-pro` | Google |
| `groq/llama-3.3-70b` | Groq |
| `bedrock/<model>` | AWS Bedrock |
| `vllm/<model>@<host>` | Self-hosted vLLM |
| `ollama/<model>@<host>` | Self-hosted Ollama |
| `openrouter/<vendor>/<model>` | OpenRouter |
| `custom/<name>/<model>` | Developer-registered via `register_llm_provider(...)` |

Bring your own LLM:

```python
from superdialog import register_llm_provider

register_llm_provider(
    name="internal",
    base_url="https://llm.example.com/v1",
    api_key=os.environ["LLM_KEY"],
    api_style="openai",
)
# usable as "custom/internal/llama-3-70b-tuned"
```

## Deploy anywhere

The same `DialogMachine` object drops into every host. The host varies; the
SuperDialog code is identical.

| Host | Adapter | Approx. LoC |
|---|---|---|
| **CLI** | none — `superdialog chat` or an `input()`/`print()` loop | ~5 |
| **LiveKit** | `superdialog.adapters.livekit.DialogMachineLLM` (`Agent(llm=...)` plugin) | ~6 |
| **PipeCat** | `superdialog.adapters.pipecat.make_processor(dm)` | ~2 |
| **FastAPI** | direct `/turn` route, or a `SessionWorker` for multi-user | ~6 |
| **Unpod Voice** | `superdialog.adapters.websocket.WebSocketRunner` | ~6 |
| **Slack / Discord / IRC / etc.** | none — direct callback | ~3 |

```python
# LiveKit — same dialog_machine, ~6 lines
from livekit.agents import Agent, AgentSession
from superdialog.adapters.livekit import DialogMachineLLM

async def entrypoint(ctx):
    agent = Agent(llm=DialogMachineLLM(dialog_machine))
    await AgentSession().start(agent=agent, room=ctx.room)
```

## Sessions

A live `DialogMachine` holds its conversation in memory for the lifetime of the
instance. When a conversation must outlive the process (multi-worker FastAPI,
day-long chat), wrap it in a `SessionWorker`:

```python
from superdialog import DialogMachine, SessionWorker, InMemorySessionStore

worker = SessionWorker(
    agent_factory=lambda: DialogMachine(flow=flow, llm="openai/gpt-5.1"),
    store=InMemorySessionStore(),
)

async with worker.acquire("user-42") as h:
    reply = await h.turn("hello")
```

The worker multiplexes N concurrent sessions, shares the immutable `Flow` by
reference, and serializes same-session requests via a per-session lock.
`LLMAgent` and `LangChainAgent` are drop-in non-state-machine brains for the
same machinery.

## CLI

| Command | Purpose |
|---|---|
| `superdialog chat <flow.json>` | Interactive terminal chat |
| `superdialog flow lint <flow.json>` | Validate graph structure |
| `superdialog flow draw <flow.json>` | Render a Mermaid diagram |
| `superdialog flow generate "<prompt>"` | Bootstrap a `flow.json` from a prompt |

## Feature status

| Capability | Status |
|---|---|
| Prompt → flow, turn execution, model URIs | ✅ v0.1 |
| Tools (Python / HTTP / MCP), `FlowSet` + `switch_flow` | ✅ v0.1 |
| CLI (`chat`, `flow lint / draw / generate`) | ✅ v0.1 |
| Adapters (LiveKit, PipeCat, FastAPI, WebSocket) | ✅ v0.1 |
| `SessionWorker` — multi-conversation lifecycle + persistence | ✅ v0.2 |
| `LLMAgent`, `LangChainAgent` — non-DM brains | ✅ v0.2 |
| `assist(text)` — mid-conversation system injection | ✅ v0.2 |
| Distributed stores (Redis / File / SQLite) + `RedisLockBackend` | 🔜 v0.3 |
| `Eval` harness + `superdialog eval` CLI | 🔜 v0.3 |
| True provider-level streaming inference | 🔜 v0.4 |

## What it is not

- **Not a UI flow designer** — that belongs to a downstream tool.
- **Not a voice framework** — audio, STT, TTS are out of scope.
- **Not multi-modal** — text only at the interface (vision/audio via tools).
- **Not a hosted service** — a library. Hosting is offered by the Unpod Voice
  Platform for those who want it.

## Documentation

| Doc | Contents |
|---|---|
| [docs/00-overview.md](docs/00-overview.md) | What it is, why standalone, positioning, roadmap |
| [docs/01-architecture.md](docs/01-architecture.md) | Components, contracts, data shapes |
| [docs/02-api-reference.md](docs/02-api-reference.md) | Every class and method |
| [docs/03-embedding-guides.md](docs/03-embedding-guides.md) | Host-by-host integration walkthroughs |

## License

Apache-2.0. See [LICENSE](LICENSE).
