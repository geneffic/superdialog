# Developer Setup Design (v2 — async-first, CLI-polish)

## Goal

Fix the broken developer onboarding experience in 3 targeted changes — no new abstractions, no sync wrappers.

## Root Cause

The library is correctly async (`create_dialog_flow`, `DialogMachine.turn`, `DialogMachine.start` are all `async def`). The docs show them as sync. Copy-pasting from docs crashes at runtime. The CLI has two rough edges that prevent the 4-line shell quick-start from working.

## Decision: Keep Async-First

Adding `asyncio.run()` sync wrappers would crash when called from inside FastAPI routes, LiveKit handlers, or PipeCat pipelines — which already run their own event loops. Same pattern as OpenAI SDK (`AsyncOpenAI` vs `OpenAI`): keep async clean, let the CLI handle the sync entry point.

---

## Scope — 3 Changes Only

### Change 1: Fix `superdialog flow generate`

**Current:** Takes positional `prompt` string, dumps JSON to stdout.
```bash
superdialog flow generate "My agent prompt"   # stdout dump, no file save
```

**Fixed:** Takes `--from <file>` OR positional `prompt`, saves to `--output <path>`.
```bash
superdialog flow generate "My agent prompt" --output flow.json
superdialog flow generate --from description.txt --output flow.json
```

**What changes in `cli/main.py`:**
- Add `--output` flag (default `flow.json`)
- Add `--from` flag as alternative to positional prompt
- Write to file instead of stdout
- Load `.env` via `load_dotenv()` before calling LLM

### Change 2: Polish `superdialog chat`

**Current:** Requires explicit positional `flow` path. No `.env` loading. Bare output (`> ` prompt, raw `print(result.text)`).

**Fixed:** `flow` becomes optional `--flow` flag defaulting to `./flow.json`. Auto-loads `.env`. Exits with clear message if flow not found.

```bash
superdialog chat                        # auto-finds ./flow.json
superdialog chat --flow my_flow.json    # explicit path still works
```

**What changes in `cli/main.py`:**
- `flow` positional → `--flow` optional (default `flow.json`)
- Add `load_dotenv()` call at top of `_cmd_chat`
- Add clear error: "No flow found. Run: superdialog flow generate --output flow.json"

### Change 3: Fix docs

Fix all sync examples in:
- `docs/README.md`
- `docs/02-api-reference.md`
- `docs/03-embedding-guides.md`

Sync example (broken):
```python
flow = create_dialog_flow(prompt="...", llm="openai/gpt-4o-mini")
reply = dm.turn("hello")
```

Async example (correct):
```python
import asyncio
from superdialog import create_dialog_flow, DialogMachine

async def main():
    flow = await create_dialog_flow(prompt="...", llm="openai/gpt-4o-mini")
    dm = DialogMachine(flow=flow, llm="openai/gpt-4o-mini")
    while True:
        user = input("> ")
        if user in ("quit", "exit"): break
        print((await dm.turn(user)).text)

asyncio.run(main())
```

---

## Developer Experience After Fix

**Shell (4 lines, zero Python):**
```bash
pip install superdialog
export OPENAI_API_KEY=sk-...
superdialog flow generate "Golf tee-time agent named Arjun" --output flow.json
superdialog chat
```

**Python script (honest, copy-pasteable):**
```python
import asyncio
from superdialog import create_dialog_flow, DialogMachine

async def main():
    flow = await create_dialog_flow(prompt=open("desc.txt").read(), llm="openai/gpt-4o-mini")
    dm = DialogMachine(flow=flow, llm="openai/gpt-4o-mini")
    while True:
        user = input("> ")
        if user in ("quit", "exit"): break
        print((await dm.turn(user)).text)

asyncio.run(main())
```

---

## Not In Scope

- `quick_start()` function — not needed
- `setup_utils.py` — not needed
- Sync wrappers — explicitly rejected (footgun in async hosts)
- Rich terminal output changes — existing bare CLI is fine
- `python-dotenv` in library deps — only CLI needs it; add to `dev` extras or call it a CLI concern
