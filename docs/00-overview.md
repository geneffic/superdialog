# SuperDialog — Overview

**Status:** Canonical
**Parent:** [README.md](README.md)

---

## 1. What it is

A Python library that turns a prompt or a flow graph into an executable dialog state machine. Pure text in, pure text out. Plays the role of the "brain" in conversational systems.

## 2. Why standalone

Two reasons:

**(a) The brain has natural reuse beyond voice.** A dialog state machine that runs a customer-onboarding flow works the same whether the user is on a phone, a WhatsApp thread, an Intercom widget, or a CLI test harness. Coupling it to telephony forecloses every non-voice use case.

**(b) The dependency direction matters.** Voice Infrastructure should depend on SuperDialog (as one brain option), not the other way around. Putting SuperDialog inside the platform makes the platform non-modular and the framework non-portable.

> *"उस machine को release करने का more less idea यह है... इस architecture से इन इस infrastructure से उसका कोई लेना देना नहीं है."*

## 3. Why OSS

- **Community pull.** LiveKit and PipeCat owe their adoption to OSS. Releasing a strong dialog framework — with good docs, working LiveKit/PipeCat adapters, and a CLI chatbot mode for evaluation — creates a top-of-funnel that no closed product can match.
- **Lower support burden.** Developers who build complex flows will keep modifying them. If the framework is theirs to fork, our team is not in the loop for every prompt change.
- **Trust.** Buyers who don't want vendor lock-in see an open core and engage further. The closed parts (telephony, voice profiles) are the parts they don't care about owning.

## 4. Why it ships first

Three reasons:

**(a) It already exists.** The dialog state machine code is the most mature part of the Unpod stack. Polishing it for OSS release is faster than building new telephony infrastructure.

**(b) Independent shippability.** It needs no telephony, no speech, no media server, no Room — none of the platform pieces. Therefore nothing on the platform side gates it.

**(c) Validation channel.** Public release is the cheapest way to learn whether the framework actually solves the *"developer wants to own their flow"* problem we hypothesize. If the OSS adoption signal is weak, the Voice Infra GTM (which depends on the same hypothesis) needs rethinking before we burn cycles on it.

## 5. Positioning

SuperDialog is to **conversation flow** what n8n is to **integration workflow** — a simple, composable, eval-able runtime for orchestrating turn-by-turn logic. Where LangChain and LangGraph expose agent primitives, SuperDialog focuses narrowly on the conversational state machine: who speaks next, what flow to switch to, when to call a tool, when to escalate.

It is intentionally smaller than LangChain in surface area. The pitch is: *"if your problem is conversation state, this is the right size."*

## 6. Audiences

| Audience | Why they care |
|---|---|
| **Voice developer using LiveKit / PipeCat today** | Drop SuperDialog in as the brain; stop hand-writing turn logic |
| **Chatbot developer (text-only)** | Use SuperDialog directly with FastAPI; test as a CLI chat |
| **Enterprise dev with their own LLM** | Plug their custom LLM URI (`custom/internal/...`) and get the rest of the framework for free |
| **Unpod Voice Infra customer** | SuperDialog is the default brain Unpod offers; same code runs locally and inside Unpod cloud |

## 7. What it does well

| Capability | Status |
|---|---|
| Prompt → flow: `await create_dialog_flow(prompt=..., llm=...)` | shipped (v0.1) |
| Turn execution: `await dialog_machine.turn(text)` | shipped (v0.1) |
| LLM provider abstraction (model URIs) | shipped (v0.1) |
| Tools: Python callables, HTTP endpoints, MCP servers | shipped (v0.1) |
| Mid-conversation flow switching (`FlowSet`, `switch_flow`) | shipped (v0.1) |
| CLI: `chat`, `flow lint / draw / generate` | shipped (v0.1) |
| Adapters: LiveKit `DialogMachineLLM`, PipeCat `make_processor`, FastAPI, WebSocket | shipped (v0.1) |
| `Agent` Protocol + `Session` + `SessionWorker` (multi-conversation lifecycle, in-process persistence, per-session locking) | shipped (v0.2) |
| `LLMAgent`, `LangChainAgent` (non-DM brains usable in SessionWorker) | shipped (v0.2) |
| `assist(text)` (renamed from `inject_system`) | shipped (v0.2) |
| Distributed stores (`RedisSessionStore`, `FileSessionStore`, `SQLiteSessionStore`) + `RedisLockBackend` | planned (v0.3) |
| Pluggable HTTP auth (`BearerAuth`, `BasicAuth`, callable) | planned (v0.3) |
| `Eval` harness + `superdialog eval` CLI | planned (v0.3) |
| True provider-level streaming inference | planned (v0.4) |
| Streaming-interruption (`UserChunk`/`AgentChunk` protocol) | planned (v0.4) |

## 8. What it explicitly is not

- **Not a UI flow designer.** That belongs to a downstream tool (future, n8n-style).
- **Not a voice framework.** Audio/STT/TTS are out of scope.
- **Not multi-modal.** Text only at the interface. (Vision/audio inputs through tools, if needed.)
- **Not a hosted service.** A library. Hosting is offered by Voice Infra for those who want it.

## 9. Success criteria

- **GitHub stars and forks.** Baseline target TBD, but real numbers — not vanity metrics.
- **Adapter usage.** Are developers actually plugging SuperDialog into LiveKit and PipeCat? Telemetry from optional usage pings if they opt in.
- **Eval adoption.** Are developers running the eval harness, or just using the runtime? The eval is part of what differentiates this from "yet another agent framework."
- **Issue and PR volume.** OSS health.
- **Unpod Voice Infra trial conversion.** Of the OSS users who try Voice Infra, what fraction stick? This is the funnel justification for releasing the framework freely.

## 10. Anti-goals

We will refuse to:
- Add features that only matter on a phone call (audio handling, RTP, SIP, etc.).
- Tie OSS releases to Unpod account creation.
- Use the OSS as a freemium ladder where critical features are paid. The framework is fully usable without ever paying Unpod.

The paid product is the Speech Pipe. The framework is the loss leader.
