# SuperDialog — OSS Dialog State Machine Framework

**Status:** Draft (product folder)
**Parent:** [../00-two-products.md](../00-two-products.md)

SuperDialog is a **standalone open-source framework** for building dialog state machines. Text in, text out. Embeddable anywhere — LiveKit, PipeCat, FastAPI, CLI, custom. Independent of Unpod's voice infrastructure.

This folder is the product specification.

---

## Contents

| Doc | Purpose |
|---|---|
| [00-overview.md](00-overview.md) | What SuperDialog is, why standalone, why OSS, why it ships first |
| [01-architecture.md](01-architecture.md) | Internals — flow graph, runtime, LLM URI resolver, tool registry, eval harness, CLI |
| [02-api-reference.md](02-api-reference.md) | Function signatures and worked examples |
| [03-embedding-guides.md](03-embedding-guides.md) | How to embed in LiveKit, PipeCat, FastAPI, CLI chatbot, unit tests |
| [decisions.md](decisions.md) | OSS-specific decisions: license, repo, governance, roadmap |

---

## TL;DR

```python
import asyncio
from superdialog import create_dialog_flow, DialogMachine, MCPTool

async def main():
    flow = await create_dialog_flow(
        prompt="Confirm KYC. Ask for Aadhaar last 4 digits.",
        llm="openai/gpt-5.1",                 # used once at construction
    )

    dialog_machine = DialogMachine(
        flow=flow,
        llm="anthropic/claude-opus-4-7",      # runtime model
        tools=[MCPTool(
            id="kerali", name="kerali", description="kerali MCP tools",
            server="https://mcp.kerali.io",
        )],
    )

    # Pure text. No infra, no phones, no sockets.
    reply = await dialog_machine.turn("मेरा Aadhaar 1234 से शुरू होता है")
    print(reply.text)

asyncio.run(main())
```

That's the entire product surface for the standalone case. Embedding into LiveKit, PipeCat, or Unpod Voice Infra is one more line in each case — see [03-embedding-guides.md](03-embedding-guides.md).

---

## What this is NOT

- **Not a hosted service.** It's a Python library you pip install.
- **Not a voice framework.** It does not handle audio, STT, or TTS.
- **Not coupled to Unpod.** You can use it without ever creating an Unpod account.
- **Not a flow UI.** It accepts prompts or pre-built flow graphs; designing flows in a visual editor is a downstream tool (n8n-style, future).
