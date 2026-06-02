# Developer Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 3 broken developer onboarding touchpoints: `flow generate` saves to file, `chat` auto-detects flow.json, docs show correct async examples.

**Architecture:** Only `src/superdialog/cli/main.py` and docs files change. No new files. No new abstractions. Library internals untouched.

**Tech Stack:** argparse (stdlib), python-dotenv, existing `create_dialog_flow` / `Flow` / `DialogMachine` APIs.

---

## File Map

| File | Action | What changes |
|---|---|---|
| `src/superdialog/cli/main.py` | Modify | `flow generate` saves to file; `chat` auto-detect + .env |
| `pyproject.toml` | Modify | Add `python-dotenv>=1.0` to dependencies |
| `docs/README.md` | Modify | Fix sync → async examples |
| `docs/02-api-reference.md` | Modify | Fix sync → async examples |
| `docs/03-embedding-guides.md` | Modify | Fix sync → async examples |
| `tests/cli/test_chat.py` | Modify | Tests for new CLI behaviour |

---

## Task 1: Add python-dotenv dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependency**

In `pyproject.toml`, change:
```toml
dependencies = [
  "pydantic>=2.5",
  "transitions>=0.9",
  "litellm>=1.50",
  "httpx>=0.27",
  "typing-extensions>=4.10",
]
```
to:
```toml
dependencies = [
  "pydantic>=2.5",
  "transitions>=0.9",
  "litellm>=1.50",
  "httpx>=0.27",
  "typing-extensions>=4.10",
  "python-dotenv>=1.0",
]
```

- [ ] **Step 2: Reinstall**

```bash
pip install -e .
```
Expected: installs cleanly.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add python-dotenv dependency for CLI .env auto-loading"
```

---

## Task 2: Fix `superdialog flow generate`

**Files:**
- Modify: `src/superdialog/cli/main.py`
- Modify: `tests/cli/test_chat.py`

Current `_cmd_generate` dumps JSON to stdout and takes a positional `prompt`. Fix: save to `--output` file, accept `--from <file>` as description input, load `.env`.

- [ ] **Step 1: Write failing tests**

Append to `tests/cli/test_chat.py`:

```python
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


def _make_mock_flow(nodes=2):
    mock = MagicMock()
    mock.model_dump.return_value = {"id": "test", "nodes": []}
    mock.nodes = [MagicMock(edges=[]) for _ in range(nodes)]
    return mock


def test_flow_generate_saves_to_output_file(tmp_path, capsys):
    mock_flow = _make_mock_flow()
    out = tmp_path / "flow.json"

    with patch("superdialog.cli.main.asyncio.run", return_value=mock_flow):
        rc = main(["flow", "generate", "A booking agent", "--output", str(out)])

    assert rc == 0
    mock_flow.save.assert_called_once_with(str(out))
    out_text = capsys.readouterr().out
    assert "Saved" in out_text or str(out) in out_text


def test_flow_generate_from_file(tmp_path, capsys):
    desc = tmp_path / "desc.txt"
    desc.write_text("A golf tee-time booking agent")
    mock_flow = _make_mock_flow()
    out = tmp_path / "flow.json"

    with patch("superdialog.cli.main.asyncio.run", return_value=mock_flow):
        rc = main(["flow", "generate", "--from", str(desc), "--output", str(out)])

    assert rc == 0
    mock_flow.save.assert_called_once_with(str(out))


def test_flow_generate_from_file_not_found(capsys):
    rc = main(["flow", "generate", "--from", "/nonexistent/desc.txt"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_flow_generate_default_output_is_flow_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    mock_flow = _make_mock_flow()

    with patch("superdialog.cli.main.asyncio.run", return_value=mock_flow):
        rc = main(["flow", "generate", "A simple agent"])

    assert rc == 0
    mock_flow.save.assert_called_once_with("flow.json")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/cli/test_chat.py::test_flow_generate_saves_to_output_file -v
```
Expected: FAIL — current `flow generate` ignores `--output`, dumps to stdout.

- [ ] **Step 3: Replace `_cmd_generate` in `cli/main.py`**

Find and replace the existing `_cmd_generate` function:

```python
def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate a flow JSON from a natural-language prompt or description file."""
    import asyncio as _asyncio
    from dotenv import load_dotenv
    from pathlib import Path as _Path

    load_dotenv()

    # Resolve description text
    from_file = getattr(args, "from_file", None)
    if from_file:
        p = _Path(from_file)
        if not p.exists():
            print(f"Error: description file not found: {from_file}", file=sys.stderr)
            return 1
        prompt = p.read_text()
    else:
        prompt = args.prompt

    if not prompt or not prompt.strip():
        print("Error: provide a prompt or --from <file>", file=sys.stderr)
        return 1

    output = getattr(args, "output", "flow.json") or "flow.json"
    llm = getattr(args, "llm", "openai/gpt-4o-mini") or "openai/gpt-4o-mini"

    print(f"Generating flow using {llm}...", flush=True)
    flow = _asyncio.run(create_dialog_flow(prompt=prompt.strip(), llm=llm))
    flow.save(output)

    node_count = len(flow.nodes)
    edge_count = sum(len(n.edges) for n in flow.nodes)
    print(f"Saved: {output}  ({node_count} nodes, {edge_count} edges)")
    return 0
```

- [ ] **Step 4: Update the `flow generate` subparser in `_build_parser`**

Find the existing `generate` sub-parser block inside `_build_parser` and replace it:

```python
    generate = flow_sub.add_parser(
        "generate", help="Generate a flow JSON from a natural-language prompt"
    )
    generate.add_argument(
        "prompt", nargs="?", default=None,
        help="Inline description string (omit if using --from)"
    )
    generate.add_argument(
        "--from", dest="from_file", metavar="FILE",
        help="Path to description file (alternative to positional prompt)"
    )
    generate.add_argument(
        "--output", default="flow.json",
        help="Output path for flow JSON (default: flow.json)"
    )
    generate.add_argument("--llm", default="openai/gpt-4o-mini")
    generate.set_defaults(fn=_cmd_generate)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/cli/test_chat.py -v
```
Expected: all pass (new + existing).

- [ ] **Step 6: Commit**

```bash
git add src/superdialog/cli/main.py tests/cli/test_chat.py
git commit -m "feat: flow generate saves to --output file and accepts --from <file>"
```

---

## Task 3: Polish `superdialog chat`

**Files:**
- Modify: `src/superdialog/cli/main.py`
- Modify: `tests/cli/test_chat.py`

Change `flow` from required positional to optional `--flow` flag (default `./flow.json`). Add `.env` loading. Add clear error when flow not found.

- [ ] **Step 1: Write failing tests**

Append to `tests/cli/test_chat.py`:

```python
def test_chat_auto_detects_flow_json(tmp_path, monkeypatch):
    flow_data = {
        "id": "t", "system_prompt": "s", "initial_node": "n",
        "nodes": [{"id": "n", "name": "N", "edges": [], "is_final": True}],
    }
    (tmp_path / "flow.json").write_text(json.dumps(flow_data))
    monkeypatch.chdir(tmp_path)

    with patch("superdialog.cli.main._run_chat_repl") as mock_repl:
        rc = main(["chat"])

    assert rc == 0
    mock_repl.assert_called_once()


def test_chat_explicit_flow_path(tmp_path):
    flow_data = {
        "id": "t", "system_prompt": "s", "initial_node": "n",
        "nodes": [{"id": "n", "name": "N", "edges": [], "is_final": True}],
    }
    flow_file = tmp_path / "custom.json"
    flow_file.write_text(json.dumps(flow_data))

    with patch("superdialog.cli.main._run_chat_repl") as mock_repl:
        rc = main(["chat", "--flow", str(flow_file)])

    assert rc == 0
    mock_repl.assert_called_once()


def test_chat_missing_flow_returns_1_with_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = main(["chat"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "generate" in err.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/cli/test_chat.py::test_chat_auto_detects_flow_json -v
```
Expected: FAIL — current `chat` requires positional `flow` arg.

- [ ] **Step 3: Replace `_cmd_chat` and extract `_run_chat_repl`**

Replace the existing `_cmd_chat` function in `cli/main.py`:

```python
def _run_chat_repl(flow: "Flow", llm: str) -> None:
    """Blocking interactive REPL. Separated for testability."""
    import asyncio as _asyncio
    machine = DialogMachine(flow=flow, llm=llm)

    async def _loop() -> None:
        result = await machine.start()
        if result.text:
            print(result.text)
        while True:
            try:
                user = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if user.strip() in {"quit", "exit"}:
                return
            if not user.strip():
                continue
            turn = await machine.turn(user)
            if turn.text:
                print(turn.text)

    _asyncio.run(_loop())


def _cmd_chat(args: argparse.Namespace) -> int:
    """Interactive REPL: auto-detect flow.json in cwd or use --flow path."""
    from pathlib import Path as _Path
    from dotenv import load_dotenv

    load_dotenv()

    flow_path = getattr(args, "flow", "flow.json") or "flow.json"
    if not _Path(flow_path).exists():
        print(
            f"No flow found at: {flow_path}\n"
            f"Run: superdialog flow generate --output {flow_path}",
            file=sys.stderr,
        )
        return 1

    flow = Flow.load(flow_path)
    llm = getattr(args, "llm", "openai/gpt-4o-mini") or "openai/gpt-4o-mini"
    _run_chat_repl(flow, llm)
    return 0
```

- [ ] **Step 4: Update `chat` subparser in `_build_parser`**

Replace the existing `chat` subparser block:

```python
    chat = sub.add_parser("chat", help="Interactive REPL against a flow")
    chat.add_argument(
        "--flow", default="flow.json",
        help="Path to flow JSON (default: ./flow.json)"
    )
    chat.add_argument("--llm", default="openai/gpt-4o-mini")
    chat.set_defaults(fn=_cmd_chat)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/cli/ -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/superdialog/cli/main.py tests/cli/test_chat.py
git commit -m "feat: chat auto-detects flow.json and loads .env automatically"
```

---

## Task 4: Fix docs — sync → async examples

**Files:**
- Modify: `docs/README.md`
- Modify: `docs/02-api-reference.md`
- Modify: `docs/03-embedding-guides.md`

No tests. Doc fixes only.

- [ ] **Step 1: Fix `docs/README.md` TL;DR block**

Find:
```python
# Pure text. No infra, no phones, no sockets.
reply = dialog_machine.turn("मेरा Aadhaar 1234 से शुरू होता है")
```

Replace with:
```python
import asyncio

# Pure text. No infra, no phones, no sockets.
reply = asyncio.run(dialog_machine.turn("मेरा Aadhaar 1234 से शुरू होता है"))
```

And wrap the `create_dialog_flow` call:
```python
flow = asyncio.run(create_dialog_flow(
    prompt="Confirm KYC. Ask for Aadhaar last 4 digits.",
    llm="openai/gpt-5.1",
))
```

- [ ] **Step 2: Fix `docs/02-api-reference.md` worked example**

In the "Worked example — end to end" section, find the CLI chatbot loop (step 4a):
```python
# ── 4a. Test as a CLI chatbot — no infrastructure needed ─────────────────
while True:
    user = input("> ")
    if user.strip() in {"quit", "exit"}: break
    print(dialog_machine.turn(user).text)
```

Replace with:
```python
# ── 4a. Test as a CLI chatbot — no infrastructure needed ─────────────────
import asyncio

async def main():
    while True:
        user = input("> ")
        if user.strip() in {"quit", "exit"}: break
        print((await dialog_machine.turn(user)).text)

asyncio.run(main())
```

Also fix the construction block at the top to use `asyncio.run`:
```python
flow = asyncio.run(create_dialog_flow(
    prompt="Verify customer KYC. Ask for Aadhaar last 4. Confirm DOB.",
    llm="openai/gpt-5.1",
))
```

And the `turn` method signature description — add note:
```
`turn()` is a coroutine. Call with `await` in async context, or `asyncio.run()` for scripts.
```

- [ ] **Step 3: Fix `docs/03-embedding-guides.md` CLI chatbot section**

Find the "CLI chatbot" example:
```python
flow = create_dialog_flow(prompt="Confirm KYC.", llm="openai/gpt-5.1")
dialog_machine = DialogMachine(flow=flow, llm="anthropic/claude-haiku-4-5")

while True:
    user = input("> ")
    if user.strip() in ("quit", "exit"): break
    print(dialog_machine.turn(user).text)
```

Replace with:
```python
import asyncio
from superdialog import create_dialog_flow, DialogMachine

async def main():
    flow = await create_dialog_flow(prompt="Confirm KYC.", llm="openai/gpt-5.1")
    dm = DialogMachine(flow=flow, llm="anthropic/claude-haiku-4-5")
    while True:
        user = input("> ")
        if user.strip() in ("quit", "exit"): break
        print((await dm.turn(user)).text)

asyncio.run(main())
```

- [ ] **Step 4: Verify no other sync examples remain**

```bash
grep -n "= create_dialog_flow\|= dialog_machine.turn\|= dm.turn" \
  docs/README.md docs/02-api-reference.md docs/03-embedding-guides.md
```
Expected: zero matches (all should now be `await` or `asyncio.run(...)`).

- [ ] **Step 5: Commit**

```bash
git add docs/README.md docs/02-api-reference.md docs/03-embedding-guides.md
git commit -m "docs: fix sync examples to correct async API (create_dialog_flow and turn are coroutines)"
```

---

## End-to-End Smoke Test

After all tasks complete, verify:

```bash
# Shell path (4 lines)
cd /tmp/test-dev-setup
echo "OPENAI_API_KEY=sk-..." > .env
superdialog flow generate "A simple greeter agent" --output flow.json
# Expected: "Saved: flow.json (N nodes, M edges)"

superdialog chat
# Expected: starts chat, finds flow.json automatically

# With description file
echo "A golf booking agent named Arjun" > desc.txt
superdialog flow generate --from desc.txt --output golf.json
superdialog chat --flow golf.json
```

```bash
# Error case
cd /tmp/empty-dir
superdialog chat
# Expected: exit 1 + "No flow found at: flow.json\nRun: superdialog flow generate --output flow.json"
```
