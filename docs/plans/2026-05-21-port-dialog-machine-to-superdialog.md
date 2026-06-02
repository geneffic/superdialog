# Port `dialog_machine/` to `superdialog/` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Also reference porting-library-code throughout for the Iron Rules (Copy First Then Edit; No Stubs No Incomplete Flows).

**Goal:** Port the existing `super/core/voice/dialog_machine/` (~8,300 LoC) into a standalone library `superdialog/src/superdialog/` that matches the `superdialog/docs/` spec (text-in/text-out, LLM URI scheme, three Tool types, four host adapters, streaming-interruption protocol).

**Architecture:** Hybrid port — hard-port the proven engine internals (machine, gate, composer, criteria, models, store, hooks, actions, extractor) verbatim; soft-port the public layer (new `DialogMachine` facade, `LLMProvider` protocol over litellm, first-class `PythonTool`/`HttpTool`/`MCPTool`, four thin host adapters including a LiveKit LLM plugin in the `livekit-plugins-langchain` style, `turn_stream()` with `user_turn_end_confidence` barge-in). Current `super/core/voice/dialog_machine/` is **untouched** during this port (parallel-lives strategy); a re-export shim flip is a separate later phase.

**Tech Stack:** Python 3.12, `uv`, `pydantic`, `pytransitions`, `litellm`, `httpx`, optional `livekit-agents`, `pipecat-ai`, `fastapi`, `mcp`, Apache 2.0 license.

**Scope explicitly EXCLUDED from this port:** `eval/`, `langGraph/`, `langchain/`, `adapters/simple_agent.py`, `adapters/livekit_bridge.py`, `adapters/flow_executor.py`, `adapters/livekit_adapter.py`, `adapters/langgraph_agent.py`, `adapters/toolcall_adapter.py`, `adapters/text_adapter.py`, `flows/`, `engine.py`, `tools/` subdir. These either become dead code or land in v0.2+.

**Decisions log:** See `superdialog/docs/decisions.md` and the brainstorming transcript that produced this plan (10 numbered decisions + 3 corrections: `.assist()` rename, JSON-first tool loading, `turn_stream()` is v1).

---

## Task 0: Scaffold the package

**Files:**
- Create: `superdialog/pyproject.toml`
- Create: `superdialog/src/superdialog/__init__.py`
- Create: `superdialog/src/superdialog/py.typed`
- Create: `superdialog/tests/__init__.py`
- Create: `superdialog/README.md`
- Create: `superdialog/LICENSE`

**Step 0.1: Author `pyproject.toml`**

```toml
[project]
name = "superdialog"
version = "0.1.0a0"
description = "Standalone dialog state machine framework — text in, text out."
readme = "README.md"
requires-python = ">=3.12"
license = { text = "Apache-2.0" }
authors = [{ name = "Unpod", email = "parvinder@unpod.ai" }]
dependencies = [
  "pydantic>=2.5",
  "transitions>=0.9",
  "litellm>=1.50",
  "httpx>=0.27",
  "typing-extensions>=4.10",
]

[project.optional-dependencies]
livekit = ["livekit-agents>=0.12"]
pipecat = ["pipecat-ai>=0.0.50"]
fastapi = ["fastapi>=0.110", "uvicorn>=0.27"]
ws = ["websockets>=12"]
mcp = ["mcp>=0.9"]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "anyio>=4", "ruff>=0.4", "pyrefly>=0.1"]

[project.scripts]
superdialog = "superdialog.cli.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/superdialog"]

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 0.2: Create stub `__init__.py`**

```python
"""SuperDialog — standalone dialog state machine framework."""
__version__ = "0.1.0a0"
```

**Step 0.3: Empty marker files**

Create `py.typed` (empty), `tests/__init__.py` (empty), copy Apache 2.0 text into `LICENSE`, write 3-line `README.md` pointing at `docs/`.

**Step 0.4: Install the workspace member**

Run: `cd superdialog && uv sync --all-extras`
Expected: resolves cleanly, creates `.venv` if missing.

**Step 0.5: Smoke check**

Run: `cd superdialog && uv run python -c "import superdialog; print(superdialog.__version__)"`
Expected: `0.1.0a0`

**Step 0.6: Commit**

```bash
git add superdialog/pyproject.toml superdialog/src superdialog/tests superdialog/README.md superdialog/LICENSE
git commit -m "chore(superdialog): scaffold package with pyproject and stub init"
```

---

## Task 1: Hard-port the Flow data model

**Files:**
- Copy: `super/core/voice/livekit/livekit_flows/core/models.py` → `superdialog/src/superdialog/flow/models.py`
- Create: `superdialog/src/superdialog/flow/__init__.py`
- Create: `superdialog/src/superdialog/flow/loader.py`
- Create: `superdialog/tests/flow/test_models.py`
- Create: `superdialog/tests/flow/test_loader.py`

**Step 1.1: Copy verbatim**

Run: `cp super/core/voice/livekit/livekit_flows/core/models.py superdialog/src/superdialog/flow/models.py`

**Step 1.2: Edit the copy — remove super.* imports**

Delete the line `from super.core.voice.dialog_machine.models import ToolDefinition`. Inline a placeholder `ToolDefinition` class at the top of the file (the spec keeps the JSON schema; we'll replace this with the unified Tool ABC in Task 4).

Inline placeholder (use exact field list from the current ToolDefinition — read `super/core/voice/dialog_machine/models.py` to get the fields):

```python
class ToolDefinition(BaseModel):
    """Placeholder — replaced by Tool ABC in superdialog.tools in Task 4."""
    id: str
    name: str
    description: str
    handler_id: str | None = None
    input_schema: dict[str, Any] | None = None
    type: str = "python"  # new discriminator field; defaults preserve current JSON
    url: str | None = None
    method: str | None = None
    server: str | None = None
```

**Step 1.3: Write tests for round-tripping existing flow JSONs**

Pick three real flow JSONs from `super/core/voice/livekit/livekit_flows/` (or wherever production flows live) and write:

```python
import json
from pathlib import Path
from superdialog.flow.models import ConversationFlow

@pytest.mark.parametrize("fixture", [
    "fixtures/kyc.json",
    "fixtures/appointment.json",
    "fixtures/escalation.json",
])
def test_existing_flow_roundtrip(fixture):
    raw = json.loads(Path(fixture).read_text())
    flow = ConversationFlow.model_validate(raw)
    # All declared fields survive the trip
    redumped = flow.model_dump(exclude_unset=True, by_alias=True)
    assert redumped == raw
```

Copy three representative flow JSONs into `superdialog/tests/fixtures/flow/` first.

**Step 1.4: Run the tests**

Run: `cd superdialog && uv run pytest tests/flow/test_models.py -v`
Expected: PASS for all three fixtures.

**Step 1.5: Add `Flow.save/load/FlowSet` in `loader.py`**

```python
from __future__ import annotations
import json
from pathlib import Path
from .models import ConversationFlow as Flow  # spec alias

class FlowSet:
    def __init__(self, flows: dict[str, Flow]):
        self.flows = flows
    def __getitem__(self, name: str) -> Flow:
        return self.flows[name]
    def names(self) -> list[str]:
        return list(self.flows.keys())

def save_flow(flow: Flow, path: str | Path) -> None:
    Path(path).write_text(json.dumps(flow.model_dump(exclude_unset=True), indent=2))

def load_flow(path: str | Path) -> Flow:
    return Flow.model_validate(json.loads(Path(path).read_text()))

# Attach as methods for spec-compatible API
Flow.save = lambda self, path: save_flow(self, path)
Flow.load = staticmethod(load_flow)
```

**Step 1.6: Add `from_prompt` bootstrap stub**

Create `superdialog/src/superdialog/flow/bootstrap.py`:

```python
"""create_dialog_flow — one-shot LLM bootstrap of a flow graph from a prompt."""
# Placeholder — full implementation lands in Task 5 after LLM resolver exists.
async def create_dialog_flow(prompt: str, llm: str, **kwargs):
    raise NotImplementedError("Implemented in Task 5")
```

**NOTE:** This `NotImplementedError` is acceptable ONLY because it gets replaced in Task 5. The porting skill forbids stubs in shipped code — Task 5 MUST replace this before the v0.1 acceptance check.

**Step 1.7: Wire the public exports**

Edit `superdialog/src/superdialog/flow/__init__.py`:

```python
from .models import ConversationFlow as Flow, FlowNode, Edge, CustomAction
from .loader import FlowSet, save_flow, load_flow
from .bootstrap import create_dialog_flow

__all__ = ["Flow", "FlowNode", "Edge", "CustomAction", "FlowSet", "create_dialog_flow"]
```

**Step 1.8: Run all flow tests**

Run: `cd superdialog && uv run pytest tests/flow/ -v`
Expected: all green.

**Step 1.9: Commit**

```bash
git add superdialog/src/superdialog/flow superdialog/tests/flow superdialog/tests/fixtures
git commit -m "feat(superdialog): hard-port Flow model + add FlowSet/save/load"
```

---

## Task 2: Hard-port the engine core (no LLM yet)

**Files:**
- Copy each file individually with `cp`; rewrite imports only.
- Create: `superdialog/src/superdialog/machine/__init__.py`

Files to copy (preserve names):

| Source | Target |
|---|---|
| `super/core/voice/dialog_machine/machine.py` | `superdialog/src/superdialog/machine/machine.py` |
| `super/core/voice/dialog_machine/gate.py` | `superdialog/src/superdialog/machine/gate.py` |
| `super/core/voice/dialog_machine/criteria.py` | `superdialog/src/superdialog/machine/criteria.py` |
| `super/core/voice/dialog_machine/models.py` | `superdialog/src/superdialog/machine/models.py` |
| `super/core/voice/dialog_machine/store.py` | `superdialog/src/superdialog/machine/store.py` |
| `super/core/voice/dialog_machine/hooks.py` | `superdialog/src/superdialog/machine/hooks.py` |
| `super/core/voice/dialog_machine/actions.py` | `superdialog/src/superdialog/machine/actions.py` |
| `super/core/voice/dialog_machine/extractor.py` | `superdialog/src/superdialog/machine/extractor.py` |
| `super/core/voice/dialog_machine/transitions.py` | `superdialog/src/superdialog/machine/transitions.py` |
| `super/core/voice/dialog_machine/runner.py` | `superdialog/src/superdialog/machine/runner.py` |
| `super/core/voice/dialog_machine/testing/` (entire dir) | `superdialog/src/superdialog/machine/testing/` |

**Step 2.1: Bulk copy**

```bash
SRC=/Users/parvbhullar/Drives/Vault/Projects/Unpod/super/super/core/voice/dialog_machine
DST=/Users/parvbhullar/Drives/Vault/Projects/Unpod/super/superdialog/src/superdialog/machine
mkdir -p $DST
for f in machine.py gate.py criteria.py models.py store.py hooks.py actions.py extractor.py transitions.py runner.py; do
  cp $SRC/$f $DST/$f
done
cp -r $SRC/testing $DST/testing
```

**Step 2.2: Find every super.* import**

Run: `cd superdialog/src/superdialog/machine && grep -rnE "from super\.|import super\." . > /tmp/super_imports.txt && cat /tmp/super_imports.txt`

Expected outputs include:
- `from super.core.voice.dialog_machine.X` (most files)
- `from super.core.voice.livekit.livekit_flows.core.models` (machine.py, gate.py)
- `from super.core.voice.common.lang_detect import detect_language` (composer-adjacent)
- `from super.core.voice.common.services import save_failed_execution_log`
- `from super.core.voice.managers.prompt_manager import get_language_name`

**Step 2.3: Rewrite internal imports**

For each match in `/tmp/super_imports.txt`:
- `from super.core.voice.dialog_machine.<X>` → `from superdialog.machine.<X>`
- `from super.core.voice.livekit.livekit_flows.core.models` → `from superdialog.flow.models`

Use `sed -i ''` (macOS) per file or apply via Edit tool.

**Step 2.4: Inline the three external utility imports**

Create `superdialog/src/superdialog/machine/_lang_util.py`:

```python
"""Inlined replacements for super.core.voice.common.* utilities."""

def detect_language(text: str) -> str:
    """Naive heuristic — Devanagari range → 'hi', else 'en'.
    Replace with langid/fasttext later if needed."""
    if any('ऀ' <= ch <= 'ॿ' for ch in text):
        return "hi"
    return "en"

_LANG_NAMES = {"en": "English", "hi": "Hindi"}
def get_language_name(code: str) -> str:
    return _LANG_NAMES.get(code, code)

def save_failed_execution_log(*args, **kwargs) -> None:
    """No-op in OSS build. Production runtime can monkey-patch this."""
    return None
```

Rewrite the three imports across the engine to point at `_lang_util`.

**Step 2.5: Delete the langgraph integration block from `machine.py`**

In the copied `machine.py`, locate and delete:
- Lines around 247-301 (the `_langgraph_pipeline` init + `_enrich_scope` call)
- The `langgraph_config` field on `NodeScope` (models.py line ~489)
- Any related call sites

Search via: `grep -nE "(langgraph|LangGraph)" superdialog/src/superdialog/machine/*.py`
Expected after edit: zero matches.

**Step 2.6: Wire the package `__init__.py`**

```python
# superdialog/src/superdialog/machine/__init__.py
from .machine import DialogStateMachine
from .models import FlowContext, TurnResult, NodeScope, TransitionRecord, ToolDescriptor, CriteriaResult
from .store import ContextStore, InMemoryContextStore
from .runner import create_machine, run_flow
from .gate import TransitionGate
from .criteria import CriteriaJudge

__all__ = ["DialogStateMachine", "FlowContext", "TurnResult", "NodeScope", "TransitionRecord",
           "ToolDescriptor", "CriteriaResult", "ContextStore", "InMemoryContextStore",
           "create_machine", "run_flow", "TransitionGate", "CriteriaJudge"]
```

**Step 2.7: Type-check**

Run: `cd superdialog && uv run pyrefly check src/superdialog/machine`
Fix any resulting errors. Expected: green.

**Step 2.8: Verify no stale imports remain**

Run: `grep -rnE "from super\.|import super\." superdialog/src/superdialog`
Expected: zero matches.

**Step 2.9: Smoke import test**

```python
# superdialog/tests/machine/test_imports.py
from superdialog.machine import (
    DialogStateMachine, FlowContext, TurnResult, NodeScope,
    InMemoryContextStore, CriteriaJudge, TransitionGate,
)

def test_engine_imports():
    assert DialogStateMachine is not None
    store = InMemoryContextStore()
    assert store is not None
```

Run: `cd superdialog && uv run pytest tests/machine/test_imports.py -v`
Expected: PASS.

**Step 2.10: Commit**

```bash
git add superdialog/src/superdialog/machine superdialog/tests/machine/test_imports.py
git commit -m "feat(superdialog): hard-port engine core; strip super.* imports and langgraph"
```

---

## Task 3: Consolidate composer + fix GAP-4

**Files:**
- Copy: `super/core/voice/dialog_machine/composer.py` → `superdialog/src/superdialog/machine/composer.py`
- Test: `superdialog/tests/machine/test_composer_language.py`

**Step 3.1: Copy**

```bash
cp super/core/voice/dialog_machine/composer.py superdialog/src/superdialog/machine/composer.py
```

**Step 3.2: Rewrite imports in the copy** (same pattern as Task 2.3)

**Step 3.3: Write a failing test for the new consolidated API**

```python
# superdialog/tests/machine/test_composer_language.py
import pytest
from superdialog.machine.composer import select_language_content

def test_picks_marked_content_for_active_language():
    text = "[EN] Hello\n[HI] नमस्ते"
    assert select_language_content(text, language="en") == "Hello"
    assert select_language_content(text, language="hi") == "नमस्ते"

def test_missing_marker_warns_not_silently_falls_back():
    text = "[EN] Hello only"
    with pytest.warns(UserWarning, match="missing marker for 'hi'"):
        result = select_language_content(text, language="hi", fallback="warn")
    assert result == "Hello only"  # falls back to English content but warns

def test_strict_mode_raises_on_missing_marker():
    text = "[EN] Hello only"
    with pytest.raises(ValueError, match="missing marker for 'hi'"):
        select_language_content(text, language="hi", fallback="raise")

def test_unmarked_content_passes_through():
    assert select_language_content("plain text", language="en") == "plain text"
    assert select_language_content("plain text", language="hi") == "plain text"
```

Run: `cd superdialog && uv run pytest tests/machine/test_composer_language.py -v`
Expected: FAIL (function not defined).

**Step 3.4: Implement the consolidated function**

Add to the top of `composer.py`:

```python
import re
import warnings
from typing import Literal

_MARKER_PATTERN = re.compile(r"\[([A-Z]{2})\]\s*(.+?)(?=\n\[[A-Z]{2}\]|\Z)", re.DOTALL)

def select_language_content(
    text: str,
    language: str,
    fallback: Literal["warn", "english", "raise"] = "warn",
) -> str:
    """Single source of truth for language marker filtering.

    Replaces filter_language_markers(), extract_speech_text(),
    process_text(), and SimpleFlowAgent._speak_node()'s inline copy.
    """
    matches = list(_MARKER_PATTERN.finditer(text))
    if not matches:
        return text  # unmarked content — pass through

    by_lang = {m.group(1).lower(): m.group(2).strip() for m in matches}
    if language in by_lang:
        return by_lang[language]

    if fallback == "raise":
        raise ValueError(f"composer: missing marker for '{language}' in: {text!r}")
    if fallback == "warn":
        warnings.warn(f"composer: missing marker for '{language}', falling back to English", UserWarning)

    return by_lang.get("en", text)
```

**Step 3.5: Run the new tests**

Run: `cd superdialog && uv run pytest tests/machine/test_composer_language.py -v`
Expected: all PASS.

**Step 3.6: Delete the 4 deprecated functions and rewire call sites**

In `composer.py`, find and remove (or rewrite as one-line shims for backward compat):
- `filter_language_markers(...)` → call `select_language_content(...)`
- `extract_speech_text(...)` → call `select_language_content(...)`
- `process_text(...)` → call `select_language_content(...)`

Search every other file in `superdialog/src/superdialog/machine/` for usages:
`grep -rnE "(filter_language_markers|extract_speech_text|process_text)" superdialog/src/superdialog/`

Rewrite each call site to use `select_language_content` directly.

**Step 3.7: Type-check**

Run: `cd superdialog && uv run pyrefly check src/superdialog/machine/composer.py`
Expected: green.

**Step 3.8: Commit**

```bash
git add superdialog/src/superdialog/machine/composer.py superdialog/tests/machine/test_composer_language.py
git commit -m "fix(superdialog): consolidate 4 language filters into select_language_content; closes GAP-4"
```

---

## Task 4: LLM provider layer + Tool ABC

**Files:**
- Create: `superdialog/src/superdialog/llm/__init__.py`
- Create: `superdialog/src/superdialog/llm/provider.py`
- Create: `superdialog/src/superdialog/llm/litellm_provider.py`
- Create: `superdialog/src/superdialog/llm/resolver.py`
- Create: `superdialog/src/superdialog/llm/registry.py`
- Create: `superdialog/src/superdialog/tools/__init__.py`
- Create: `superdialog/src/superdialog/tools/base.py`
- Create: `superdialog/src/superdialog/tools/python_tool.py`
- Create: `superdialog/src/superdialog/tools/http_tool.py`
- Create: `superdialog/src/superdialog/tools/mcp_tool.py`
- Test: `superdialog/tests/llm/test_resolver.py`
- Test: `superdialog/tests/tools/test_*.py`

**Step 4.1: Define the LLMProvider protocol**

`superdialog/src/superdialog/llm/provider.py`:

```python
from __future__ import annotations
from typing import Protocol, AsyncIterator, Any
from dataclasses import dataclass

@dataclass
class CompletionResult:
    text: str
    tool_calls: list[dict[str, Any]]
    metadata: dict[str, Any]  # latency_ms, prompt_tokens, completion_tokens, model

@dataclass
class StreamChunk:
    text: str | None
    tool_call_delta: dict | None
    done: bool

class LLMProvider(Protocol):
    async def complete(
        self, messages: list[dict], tools: list[dict] | None = None, **opts: Any
    ) -> CompletionResult: ...
    async def stream(
        self, messages: list[dict], tools: list[dict] | None = None, **opts: Any
    ) -> AsyncIterator[StreamChunk]: ...
```

**Step 4.2: Implement LitellmProvider**

`superdialog/src/superdialog/llm/litellm_provider.py`:

```python
from __future__ import annotations
import time
from typing import Any, AsyncIterator
import litellm
from .provider import LLMProvider, CompletionResult, StreamChunk

class LitellmProvider:
    def __init__(self, model: str, **default_opts):
        self.model = model
        self.default_opts = default_opts

    async def complete(self, messages, tools=None, **opts):
        merged = {**self.default_opts, **opts}
        t0 = time.perf_counter()
        resp = await litellm.acompletion(
            model=self.model, messages=messages, tools=tools, **merged
        )
        msg = resp.choices[0].message
        return CompletionResult(
            text=msg.content or "",
            tool_calls=[tc.model_dump() for tc in (msg.tool_calls or [])],
            metadata={
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "model": self.model,
            },
        )

    async def stream(self, messages, tools=None, **opts) -> AsyncIterator[StreamChunk]:
        merged = {**self.default_opts, **opts, "stream": True}
        async for chunk in await litellm.acompletion(
            model=self.model, messages=messages, tools=tools, **merged
        ):
            delta = chunk.choices[0].delta
            yield StreamChunk(
                text=delta.content,
                tool_call_delta=delta.tool_calls[0].model_dump() if delta.tool_calls else None,
                done=chunk.choices[0].finish_reason is not None,
            )
```

**Step 4.3: Implement the registry + resolver**

`superdialog/src/superdialog/llm/registry.py`:

```python
from dataclasses import dataclass

@dataclass
class CustomProviderConfig:
    base_url: str
    api_key: str
    api_style: str = "openai"

_REGISTRY: dict[str, CustomProviderConfig] = {}

def register_llm_provider(name: str, base_url: str, api_key: str, api_style: str = "openai") -> None:
    _REGISTRY[name] = CustomProviderConfig(base_url=base_url, api_key=api_key, api_style=api_style)

def get_custom(name: str) -> CustomProviderConfig | None:
    return _REGISTRY.get(name)
```

`superdialog/src/superdialog/llm/resolver.py`:

```python
from .provider import LLMProvider
from .litellm_provider import LitellmProvider
from .registry import get_custom

def resolve_llm(uri: str) -> LLMProvider:
    """Parse a model URI and return an LLMProvider instance.

    Examples:
        openai/gpt-5.1                          → LitellmProvider("openai/gpt-5.1")
        anthropic/claude-opus-4-7               → LitellmProvider("anthropic/claude-opus-4-7")
        custom/<name>/<model>                   → LitellmProvider with registered base_url
        vllm/<model>@<host>                     → LitellmProvider("hosted_vllm/<model>", api_base=<host>)
        ollama/<model>@<host>                   → LitellmProvider("ollama/<model>", api_base=<host>)
        unpod/<vertical>                        → LitellmProvider via custom unpod config
    """
    if uri.startswith("custom/"):
        _, name, *rest = uri.split("/", 2)
        cfg = get_custom(name)
        if not cfg:
            raise ValueError(f"Unknown custom provider: {name}")
        model = rest[0] if rest else ""
        return LitellmProvider(model=f"openai/{model}", api_base=cfg.base_url, api_key=cfg.api_key)
    if "@" in uri:
        provider_model, host = uri.split("@", 1)
        # vllm/foo@host  → hosted_vllm/foo via api_base
        scheme, model = provider_model.split("/", 1)
        litellm_scheme = {"vllm": "hosted_vllm", "ollama": "ollama"}.get(scheme, scheme)
        return LitellmProvider(model=f"{litellm_scheme}/{model}", api_base=host)
    return LitellmProvider(model=uri)
```

**Step 4.4: Test the resolver**

`superdialog/tests/llm/test_resolver.py`:

```python
import pytest
from superdialog.llm.resolver import resolve_llm
from superdialog.llm.registry import register_llm_provider
from superdialog.llm.litellm_provider import LitellmProvider

def test_openai_uri():
    p = resolve_llm("openai/gpt-5.1")
    assert isinstance(p, LitellmProvider)
    assert p.model == "openai/gpt-5.1"

def test_anthropic_uri():
    p = resolve_llm("anthropic/claude-opus-4-7")
    assert p.model == "anthropic/claude-opus-4-7"

def test_vllm_with_host():
    p = resolve_llm("vllm/llama-3@http://my-vllm:8000")
    assert p.model == "hosted_vllm/llama-3"
    assert p.default_opts.get("api_base") == "http://my-vllm:8000"

def test_custom_provider_requires_registration():
    with pytest.raises(ValueError, match="Unknown custom provider"):
        resolve_llm("custom/unknown/model")

def test_custom_provider_after_registration():
    register_llm_provider("kerali", "https://llm.kerali.io/v1", "key-123")
    p = resolve_llm("custom/kerali/llama-3-70b")
    assert isinstance(p, LitellmProvider)
    assert p.default_opts.get("api_base") == "https://llm.kerali.io/v1"
```

Run: `cd superdialog && uv run pytest tests/llm/ -v`
Expected: all PASS.

**Step 4.5: Define the Tool ABC**

`superdialog/src/superdialog/tools/base.py`:

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolResult:
    data: dict[str, Any]
    transition_edge_id: str | None = None
    error: str | None = None

class Tool(ABC):
    id: str
    name: str
    description: str
    input_schema: dict[str, Any] | None

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> ToolResult: ...

    def to_openai_function(self) -> dict[str, Any]:
        """Render this tool as an OpenAI function-tool spec."""
        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }

    @staticmethod
    def from_dict(spec: dict[str, Any], handler_registry: dict | None = None) -> "Tool":
        """Deserialize a JSON tool entry into the right Tool subclass via `type` discriminator."""
        from .python_tool import PythonTool
        from .http_tool import HttpTool
        from .mcp_tool import MCPTool
        ttype = spec.get("type", "python")
        if ttype == "python":
            handler = (handler_registry or {}).get(spec.get("handler_id"))
            return PythonTool(
                id=spec["id"], name=spec.get("name", spec["id"]),
                description=spec.get("description", ""),
                input_schema=spec.get("input_schema"),
                fn=handler,
            )
        if ttype == "http":
            return HttpTool(
                id=spec["id"], name=spec.get("name", spec["id"]),
                description=spec.get("description", ""),
                input_schema=spec.get("input_schema"),
                url=spec["url"], method=spec.get("method", "POST"),
                auth=spec.get("auth"),
            )
        if ttype == "mcp":
            return MCPTool(
                id=spec["id"], name=spec.get("name", spec["id"]),
                description=spec.get("description", ""),
                server=spec["server"],
            )
        raise ValueError(f"Unknown tool type: {ttype}")
```

**Step 4.6: Implement PythonTool, HttpTool, MCPTool**

`superdialog/src/superdialog/tools/python_tool.py`:

```python
from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from .base import Tool, ToolResult

@dataclass
class PythonTool(Tool):
    id: str
    name: str
    description: str
    fn: Callable[..., Any] | Callable[..., Awaitable[Any]] | None = None
    input_schema: dict[str, Any] | None = None

    @classmethod
    def of(cls, fn: Callable, name: str | None = None, description: str | None = None) -> "PythonTool":
        """Convenience constructor: PythonTool.of(my_function)."""
        sig = inspect.signature(fn)
        return cls(
            id=name or fn.__name__,
            name=name or fn.__name__,
            description=description or (fn.__doc__ or "").strip(),
            fn=fn,
            input_schema=cls._infer_schema(sig),
        )

    @staticmethod
    def _infer_schema(sig: inspect.Signature) -> dict:
        props = {}
        required = []
        for pname, p in sig.parameters.items():
            ann = p.annotation
            jtype = {str: "string", int: "integer", float: "number", bool: "boolean"}.get(ann, "string")
            props[pname] = {"type": jtype}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        return {"type": "object", "properties": props, "required": required}

    async def execute(self, args: dict) -> ToolResult:
        if self.fn is None:
            return ToolResult(data={}, error=f"PythonTool {self.id!r} has no handler bound")
        result = self.fn(**args)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolResult):
            return result
        return ToolResult(data=result if isinstance(result, dict) else {"value": result})
```

`superdialog/src/superdialog/tools/http_tool.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import httpx
from .base import Tool, ToolResult

@dataclass
class HttpTool(Tool):
    id: str
    name: str
    description: str
    url: str
    method: str = "POST"
    auth: dict[str, Any] | None = None
    input_schema: dict[str, Any] | None = None

    async def execute(self, args: dict) -> ToolResult:
        headers = {}
        if self.auth and self.auth.get("type") == "bearer":
            headers["Authorization"] = f"Bearer {self.auth['token']}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.request(self.method, self.url, json=args, headers=headers)
            if r.status_code >= 400:
                return ToolResult(data={}, error=f"HTTP {r.status_code}: {r.text}")
            try:
                payload = r.json()
            except Exception:
                payload = {"raw": r.text}
            return ToolResult(data=payload if isinstance(payload, dict) else {"value": payload})
```

`superdialog/src/superdialog/tools/mcp_tool.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .base import Tool, ToolResult

@dataclass
class MCPTool(Tool):
    id: str
    name: str
    description: str
    server: str
    input_schema: dict[str, Any] | None = None
    # Connection lifecycle managed lazily; mcp client imported at execute time
    _client: Any = field(default=None, init=False, repr=False)

    async def _ensure_connected(self):
        if self._client is not None:
            return
        try:
            from mcp.client.session import ClientSession
            from mcp.client.sse import sse_client
        except ImportError as e:
            raise RuntimeError("MCPTool requires `pip install superdialog[mcp]`") from e
        # Real implementation: open SSE channel, list tools, store handles
        # Spec: MCPTool exposes ALL tools from the server under its namespace.
        # For v0.1, treat MCPTool as a single proxy: forward args to server.call(self.id, args).
        self._client = await sse_client(self.server).__aenter__()  # simplified; real impl manages context

    async def execute(self, args: dict) -> ToolResult:
        await self._ensure_connected()
        # Forward to MCP server; implementation depends on mcp client API
        # PORT NOTE: complete this against the real mcp client before v0.1 ships.
        # NO STUB — this must work end-to-end.
        from mcp.types import CallToolRequest  # type: ignore
        result = await self._client.call_tool(self.id, args)  # adjust to actual API
        return ToolResult(data=dict(result.content[0].text) if result.content else {})
```

**NOTE:** MCPTool's real implementation depends on the `mcp` client API. Step 4.6 ends with a working-but-minimal SSE call; the actual API surface needs to be verified against the live `mcp` package before this task closes. **No `pass`/`NotImplementedError`** — if the MCP API isn't what we expect, finish the implementation against what it actually is.

**Step 4.7: Test each tool**

```python
# superdialog/tests/tools/test_python_tool.py
import pytest
from superdialog.tools import PythonTool, ToolResult

async def hello(name: str) -> dict:
    return {"greeting": f"Hello, {name}"}

@pytest.mark.asyncio
async def test_python_tool_from_function():
    tool = PythonTool.of(hello)
    result = await tool.execute({"name": "World"})
    assert result.data == {"greeting": "Hello, World"}

@pytest.mark.asyncio
async def test_python_tool_infers_schema():
    tool = PythonTool.of(hello)
    assert tool.input_schema["properties"]["name"]["type"] == "string"
    assert "name" in tool.input_schema["required"]
```

```python
# superdialog/tests/tools/test_http_tool.py
import pytest
import httpx
from superdialog.tools import HttpTool

@pytest.mark.asyncio
async def test_http_tool_posts_and_parses_json(httpx_mock):
    httpx_mock.add_response(json={"ok": True, "id": 42})
    tool = HttpTool(id="lookup", name="lookup", description="", url="https://example.com/api")
    result = await tool.execute({"q": "x"})
    assert result.data == {"ok": True, "id": 42}
```

Run: `cd superdialog && uv run pytest tests/tools/ -v`
Expected: PASS.

**Step 4.8: Wire tools into the engine**

Edit `superdialog/src/superdialog/machine/machine.py`:
- Constructor: accept `tools: list[Tool]` and `handler_registry: dict[str, Callable]` (the latter for resolving JSON `handler_id` entries).
- Replace internal `tool_handlers: dict` with `tools_by_id: dict[str, Tool]`.
- `machine.execute_tool(tool_id, args)` becomes: `await self.tools_by_id[tool_id].execute(args)`.
- On `from_flow`, deserialize `flow.tools` and each `node.tools` via `Tool.from_dict(spec, handler_registry)`.

Provide a **backward-compat shim**: if callers still pass `tool_handlers={"id": fn}`, wrap them as `PythonTool(id=k, fn=v, ...)` at construction time. Mark this deprecated in a docstring; do not remove.

**Step 4.9: Rewire LLM calls in CriteriaJudge to use the provider**

In `superdialog/src/superdialog/machine/criteria.py`, change the constructor:

```python
# Before:
class CriteriaJudge:
    def __init__(self, llm_fn: LLMCallable): ...
# After:
class CriteriaJudge:
    def __init__(self, llm: "LLMProvider"): ...
        # call sites: await self.llm.complete(messages)  → returns CompletionResult
```

Update every call site in `machine.py` and `runner.py` similarly. Keep `LLMCallable` as a deprecated type alias for the parallel-lives period.

**Step 4.10: Type-check + test**

Run: `cd superdialog && uv run pyrefly check src/superdialog && uv run pytest tests/ -v`
Expected: all green.

**Step 4.11: Commit**

```bash
git add superdialog/src/superdialog/llm superdialog/src/superdialog/tools superdialog/tests/llm superdialog/tests/tools
git commit -m "feat(superdialog): LLMProvider protocol + litellm impl + Tool ABC with 3 subclasses"
```

---

## Task 5: DialogMachine facade + streaming-interruption

**Files:**
- Create: `superdialog/src/superdialog/dialog_machine.py`
- Create: `superdialog/src/superdialog/stream.py`
- Update: `superdialog/src/superdialog/__init__.py`
- Update: `superdialog/src/superdialog/flow/bootstrap.py` (replace the Task 1.6 NotImplementedError)
- Test: `superdialog/tests/test_dialog_machine.py`
- Test: `superdialog/tests/test_turn_stream.py`

**Step 5.1: Define the public types**

`superdialog/src/superdialog/stream.py`:

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class UserChunk:
    text: str
    end_confidence: float  # 0..1; producer of the stream (LiveKit VAD, etc.) supplies this

@dataclass
class AgentChunk:
    event: Literal["token", "speech_start", "paused_for_user", "tool_call", "done"]
    text: str | None = None
    tool_call: dict | None = None
    metadata: dict = field(default_factory=dict)

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

**Step 5.2: Implement the DialogMachine facade**

`superdialog/src/superdialog/dialog_machine.py`:

```python
from __future__ import annotations
import asyncio
from typing import AsyncIterator, Literal
from .flow.models import ConversationFlow as Flow
from .flow.loader import FlowSet
from .llm.resolver import resolve_llm
from .llm.provider import LLMProvider
from .tools.base import Tool
from .machine.machine import DialogStateMachine
from .machine.store import ContextStore, InMemoryContextStore
from .stream import UserChunk, AgentChunk, Turn, ToolCall

class DialogMachine:
    """Spec-aligned public facade over DialogStateMachine."""

    def __init__(
        self,
        flow: Flow | FlowSet,
        llm: str,
        tools: list[Tool] | None = None,
        memory: ContextStore | None = None,
        config: dict | None = None,
    ):
        self._flowset = flow if isinstance(flow, FlowSet) else FlowSet({"main": flow})
        self._active_flow_name = "main"
        self._llm_uri = llm
        self._llm: LLMProvider = resolve_llm(llm)
        self._tools = tools or []
        self._memory = memory or InMemoryContextStore()
        self._config = config or {}
        self._machine: DialogStateMachine | None = None
        # Defer building DialogStateMachine until first use (it's async)

    async def _ensure_machine(self):
        if self._machine is None:
            self._machine = await DialogStateMachine.from_flow(
                flow=self._flowset[self._active_flow_name],
                # adapter parameter: we now inject the LLMProvider directly;
                # see Task 4.9 — CriteriaJudge takes provider, not callable.
                adapter=None,  # No RuntimeAdapter needed in facade mode
                tools=self._tools,
                llm=self._llm,
                store=self._memory,
            )
        return self._machine

    async def turn(
        self,
        text: str,
        context: dict | None = None,
        stream: bool | Literal["text"] = False,
    ) -> Turn | AsyncIterator[AgentChunk]:
        m = await self._ensure_machine()
        if stream == "text":
            return self._stream_tokens(text, context)
        result = await m.process_turn(text)
        return Turn(
            text=result.response,
            tool_calls=[ToolCall(**tc) for tc in getattr(result, "tool_calls", [])],
            metadata={
                "from_node": result.from_node,
                "to_node": result.to_node,
                "outcome": result.outcome,
                "edge_id": result.edge_id,
            },
        )

    async def _stream_tokens(self, text: str, context):
        m = await self._ensure_machine()
        # Build LLM call and forward tokens; tool calls emerge as tool_call events.
        # Implementation: delegates to provider.stream() with the enriched instruction.
        async for chunk in m.stream_turn(text):
            yield AgentChunk(event="token", text=chunk.text)
        yield AgentChunk(event="done")

    async def turn_stream(
        self,
        chunks: AsyncIterator[UserChunk],
        end_confidence_high: float = 0.78,
        end_confidence_low: float = 0.28,
        dual_mode: bool = True,
    ) -> AsyncIterator[AgentChunk]:
        """v1 streaming-interruption protocol.

        State machine:
            BUFFERING — accumulate user text; if end_confidence > high, switch to GENERATING
            GENERATING — drive LLM; on new UserChunk:
                if dual_mode: cancel LLM task, prepend partial agent text to a holding queue,
                              return to BUFFERING with the new chunk
                else: ignore new chunks until done
        """
        m = await self._ensure_machine()
        buffer = ""
        gen_task: asyncio.Task | None = None
        state: Literal["BUFFERING", "GENERATING"] = "BUFFERING"

        agent_output_queue: asyncio.Queue[AgentChunk] = asyncio.Queue()

        async def _drive_llm(prompt: str):
            try:
                async for tok in m.stream_turn(prompt):
                    await agent_output_queue.put(AgentChunk(event="token", text=tok.text))
                await agent_output_queue.put(AgentChunk(event="done"))
            except asyncio.CancelledError:
                await agent_output_queue.put(AgentChunk(event="paused_for_user"))
                raise

        async for uc in chunks:
            if state == "BUFFERING":
                buffer += uc.text
                if uc.end_confidence >= end_confidence_high:
                    state = "GENERATING"
                    gen_task = asyncio.create_task(_drive_llm(buffer))
                    buffer = ""
                    yield AgentChunk(event="speech_start")
            elif state == "GENERATING":
                # User interrupted while agent was generating
                if dual_mode and gen_task and not gen_task.done():
                    gen_task.cancel()
                    try: await gen_task
                    except asyncio.CancelledError: pass
                    state = "BUFFERING"
                    buffer = uc.text
                # else: drop the chunk (single-mode policy)
            # Drain anything ready
            while not agent_output_queue.empty():
                yield agent_output_queue.get_nowait()

        # End of user stream — drain
        if gen_task and not gen_task.done():
            await gen_task
        while not agent_output_queue.empty():
            yield agent_output_queue.get_nowait()

    def assist(self, text: str) -> None:
        """Push a system-level steering instruction for the next turn."""
        # Stored on the machine; consumed by the next process_turn/turn_stream
        if self._machine:
            self._machine.queue_system_message(text)
        else:
            self._config.setdefault("pending_system_messages", []).append(text)

    def reset(self) -> None:
        self._machine = None

    def set_llm(self, uri: str) -> None:
        self._llm_uri = uri
        self._llm = resolve_llm(uri)
        if self._machine:
            self._machine.set_llm_provider(self._llm)

    def switch_flow(self, name: str, preserve_memory: bool = False) -> None:
        if name not in self._flowset.flows:
            raise KeyError(f"Flow {name!r} not in FlowSet")
        self._active_flow_name = name
        prev_memory = self._memory if preserve_memory else None
        self._machine = None  # rebuilt on next ensure
        if not preserve_memory:
            self._memory = InMemoryContextStore()

    @property
    def state(self) -> dict:
        if not self._machine:
            return {}
        return {
            "node_id": self._machine.context.current_node_id,
            "slots": dict(self._machine.context.userdata),
        }
```

**NOTE:** Two internal hooks referenced above (`m.stream_turn(prompt)`, `m.queue_system_message(text)`, `m.set_llm_provider(p)`) may not exist verbatim in the ported engine. **If they don't, add them as thin methods on `DialogStateMachine` in this task** — do not leave any of them as `NotImplementedError`. The porting skill's Iron Rule is in force.

**Step 5.3: Replace the Task 1.6 placeholder**

Rewrite `superdialog/src/superdialog/flow/bootstrap.py`:

```python
from __future__ import annotations
import json
from .models import ConversationFlow as Flow
from ..llm.resolver import resolve_llm

_BOOTSTRAP_SYSTEM = """You are a dialog-flow generator. Given a high-level prompt,
emit a JSON flow with system_prompt, initial_node, nodes[], edges, and
completion_criteria. Output ONLY valid JSON matching the ConversationFlow schema."""

async def create_dialog_flow(prompt: str, llm: str, **kwargs) -> Flow:
    provider = resolve_llm(llm)
    messages = [
        {"role": "system", "content": _BOOTSTRAP_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    result = await provider.complete(messages, response_format={"type": "json_object"})
    data = json.loads(result.text)
    return Flow.model_validate(data)
```

**Step 5.4: Update the top-level exports**

`superdialog/src/superdialog/__init__.py`:

```python
"""SuperDialog — standalone dialog state machine framework."""
__version__ = "0.1.0a0"

from .dialog_machine import DialogMachine
from .stream import UserChunk, AgentChunk, Turn, ToolCall
from .flow import Flow, FlowSet, create_dialog_flow
from .tools import Tool, ToolResult, PythonTool, HttpTool, MCPTool
from .llm.registry import register_llm_provider

__all__ = [
    "DialogMachine", "Flow", "FlowSet", "create_dialog_flow",
    "Tool", "ToolResult", "PythonTool", "HttpTool", "MCPTool",
    "UserChunk", "AgentChunk", "Turn", "ToolCall",
    "register_llm_provider",
]
```

**Step 5.5: Write spec-shape integration test**

```python
# superdialog/tests/test_dialog_machine.py
import os
import pytest
from superdialog import DialogMachine, Flow, PythonTool

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
@pytest.mark.asyncio
async def test_kyc_flow_text_turn():
    flow = Flow.load("tests/fixtures/flow/kyc.json")
    m = DialogMachine(flow=flow, llm="openai/gpt-4o-mini")
    result = await m.turn("Hello, my name is Alice")
    assert isinstance(result.text, str) and len(result.text) > 0
    assert "node_id" in result.metadata.get("to_node", "") or result.metadata.get("to_node")
```

```python
# superdialog/tests/test_turn_stream.py
import pytest
from superdialog import DialogMachine, Flow, UserChunk

async def _user_stream():
    yield UserChunk(text="मेरा ", end_confidence=0.1)
    yield UserChunk(text="आधार ", end_confidence=0.3)
    yield UserChunk(text="1234 है", end_confidence=0.92)

@pytest.mark.asyncio
async def test_turn_stream_triggers_generation_above_threshold(monkeypatch):
    flow = Flow.load("tests/fixtures/flow/kyc.json")
    m = DialogMachine(flow=flow, llm="openai/gpt-4o-mini")
    # Mock the underlying machine.stream_turn to avoid real LLM in this test
    async def fake_stream(prompt):
        yield type("T", (), {"text": "धन्यवाद, "})()
        yield type("T", (), {"text": "Alice।"})()
    monkeypatch.setattr(m, "_stream_tokens", lambda *a, **k: fake_stream(""))
    events = [e async for e in m.turn_stream(_user_stream())]
    assert any(e.event == "speech_start" for e in events)
    assert any(e.event == "token" for e in events)
    assert events[-1].event == "done"
```

Run: `cd superdialog && uv run pytest tests/test_turn_stream.py -v`
Expected: PASS.

**Step 5.6: Add an interruption test**

```python
@pytest.mark.asyncio
async def test_turn_stream_interrupts_in_dual_mode():
    """Verify that a fresh user chunk during GENERATING cancels the LLM task."""
    # Setup: feed a high-confidence chunk to start generation, then another chunk before done
    flow = Flow.load("tests/fixtures/flow/kyc.json")
    m = DialogMachine(flow=flow, llm="openai/gpt-4o-mini")

    interrupt_seen = False
    async def fake_stream(prompt):
        await asyncio.sleep(0.5)  # simulate slow LLM
        yield type("T", (), {"text": "should-not-appear"})()
    # ... wire monkeypatch, feed UserChunk(0.9) then UserChunk(0.5) quickly,
    # assert AgentChunk(event="paused_for_user") appears in output
```

(Flesh out concretely during execution.)

**Step 5.7: Run all tests**

Run: `cd superdialog && uv run pytest -v`
Expected: green (skipping LLM-network tests if no key).

**Step 5.8: Commit**

```bash
git add superdialog/src/superdialog superdialog/tests
git commit -m "feat(superdialog): DialogMachine facade + turn_stream() v1 with dual-mode barge-in"
```

---

## Task 6: Adapters (LiveKit, PipeCat, FastAPI, WebSocket)

**Files:**
- Create: `superdialog/src/superdialog/adapters/__init__.py`
- Create: `superdialog/src/superdialog/adapters/livekit.py`
- Create: `superdialog/src/superdialog/adapters/pipecat.py`
- Create: `superdialog/src/superdialog/adapters/fastapi.py`
- Create: `superdialog/src/superdialog/adapters/websocket.py`
- Tests: `superdialog/tests/adapters/test_*.py`
- Examples: `superdialog/examples/{livekit,pipecat,fastapi,ws}.py`

**Step 6.1: LiveKit LLM plugin** (Q7 — `livekit-plugins-langchain`-style)

Read first: `https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-langchain/livekit/plugins/langchain` to mirror the protocol.

`superdialog/src/superdialog/adapters/livekit.py`:

```python
from __future__ import annotations
from typing import Any, AsyncIterator
# Heavy livekit-agents imports are deferred so the rest of the lib works without [livekit] extra.

class DialogMachineLLM:
    """LiveKit Agent `llm=` adapter implementing livekit.agents.llm.LLM protocol.

    Usage:
        from livekit.agents import Agent, AgentSession
        from superdialog.adapters.livekit import DialogMachineLLM
        await session.start(
            agent=Agent(llm=DialogMachineLLM(dialog_machine)),
        )

    Sources confidence from LiveKit VAD/turn-detection signals and feeds
    DialogMachine.turn_stream() with UserChunk(text, end_confidence).
    """

    def __init__(self, dialog_machine, end_confidence_high: float = 0.78, end_confidence_low: float = 0.28):
        try:
            from livekit.agents import llm as lk_llm  # noqa: F401
        except ImportError as e:
            raise RuntimeError("DialogMachineLLM requires `pip install superdialog[livekit]`") from e
        self.dialog_machine = dialog_machine
        self.high = end_confidence_high
        self.low = end_confidence_low

    # Mirror the livekit.agents.llm.LLM interface. The exact method signatures
    # depend on the livekit-agents version pinned in [livekit] extra. Below is
    # a sketch; align with the actual protocol when implementing.

    def chat(self, *, chat_ctx, fnc_ctx=None, conn_options=None, **kwargs):
        from .livekit_stream import DialogMachineStream
        return DialogMachineStream(self.dialog_machine, chat_ctx, fnc_ctx, self.high, self.low)
```

`superdialog/src/superdialog/adapters/livekit_stream.py`:

```python
"""LLMStream subclass that drives DialogMachine.turn_stream() from LiveKit chat context.

Mirror livekit-plugins-langchain's stream class. Roughly:
1. Read the latest user message from chat_ctx.
2. Use the participant's VAD signal (LiveKit AgentSession state) as end_confidence.
3. Yield ChatChunk frames as DialogMachine produces AgentChunk tokens.
"""
# Full implementation goes here — no stubs allowed.
```

**Step 6.2: Test the LiveKit adapter against a real LiveKit agent**

(Integration test, skipped if `livekit-agents` not installed.)

**Step 6.3: PipeCat adapter**

`superdialog/src/superdialog/adapters/pipecat.py`:

```python
class DialogMachineProcessor:
    """PipeCat FrameProcessor that drives DialogMachine on inbound TextFrames."""
    def __init__(self, dialog_machine):
        try:
            from pipecat.processors.frame_processor import FrameProcessor
            from pipecat.frames.frames import TextFrame
        except ImportError as e:
            raise RuntimeError("DialogMachineProcessor requires `pip install superdialog[pipecat]`") from e
        # PipeCat's FrameProcessor uses inheritance; implement that pattern here.
```

Mirror the spec from `02-api-reference.md §Adapters / PipeCat`. Full implementation required.

**Step 6.4: FastAPI adapter**

```python
# superdialog/src/superdialog/adapters/fastapi.py
from fastapi import APIRouter, FastAPI

def make_router(dialog_machine) -> APIRouter:
    r = APIRouter()
    @r.post("/turn")
    async def turn(payload: dict):
        result = await dialog_machine.turn(payload["text"])
        return {"reply": result.text, "metadata": result.metadata}
    return r

class FastAPIRouter:
    def __init__(self, dialog_machine):
        self.dialog_machine = dialog_machine
        self.router = make_router(dialog_machine)
    def mount(self, app: FastAPI, prefix: str = ""):
        app.include_router(self.router, prefix=prefix)
```

**Step 6.5: WebSocket runner**

```python
# superdialog/src/superdialog/adapters/websocket.py
import json
import asyncio
import websockets

class WebSocketRunner:
    def __init__(self, dialog_machine, agent_id: str, api_key: str | None = None):
        self.dm = dialog_machine
        self.agent_id = agent_id
        self.api_key = api_key

    async def _handler(self, ws):
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "user_text":
                result = await self.dm.turn(msg["text"])
                await ws.send(json.dumps({"type": "agent_text", "text": result.text}))
            elif msg.get("type") == "user_chunk":
                # streaming-interruption protocol; flesh out per spec
                ...

    def serve(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        async def _main():
            async with websockets.serve(self._handler, host, port):
                await asyncio.Future()
        asyncio.run(_main())
```

**Step 6.6: Write one working example per adapter** in `superdialog/examples/`, plus a smoke test that imports each adapter (skipping if the extra isn't installed).

**Step 6.7: Run tests**

Run: `cd superdialog && uv run pytest tests/adapters/ -v`
Expected: green or skipped-for-missing-extras.

**Step 6.8: Commit**

```bash
git add superdialog/src/superdialog/adapters superdialog/examples superdialog/tests/adapters
git commit -m "feat(superdialog): four host adapters (livekit LLM plugin, pipecat, fastapi, ws)"
```

---

## Task 7: Port the workspace test suite

**Files:**
- Copy: `tests/core/voice/dialog_machine/` → `superdialog/tests/dialog_machine/`

**Step 7.1: Locate the source tests**

Run: `find tests -path '*dialog_machine*' -name '*.py' | head -30`
Note all test files.

**Step 7.2: Bulk copy**

```bash
cp -r tests/core/voice/dialog_machine superdialog/tests/dialog_machine
```

**Step 7.3: Rewrite imports**

In every copied file:
- `from super.core.voice.dialog_machine.X` → `from superdialog.machine.X`
- `from super.core.voice.livekit.livekit_flows.core.models` → `from superdialog.flow.models`

Search after edit: `grep -rnE "from super\.|import super\." superdialog/tests/dialog_machine`
Expected: zero matches.

**Step 7.4: Adapt constructors**

Tests that construct `DialogStateMachine.from_flow(flow, adapter)` and pass an `LLMCallable`-shaped callback need adaptation to pass an `LLMProvider` instead. Create a `tests/conftest.py`:

```python
@pytest.fixture
def fake_llm_provider():
    """Returns an LLMProvider that records calls and returns scripted responses."""
    class FakeProvider:
        def __init__(self): self.calls = []; self.scripted = []
        async def complete(self, messages, tools=None, **opts):
            self.calls.append({"messages": messages, "tools": tools})
            from superdialog.llm.provider import CompletionResult
            return self.scripted.pop(0) if self.scripted else CompletionResult(text="", tool_calls=[], metadata={})
        async def stream(self, messages, tools=None, **opts):
            yield None
    return FakeProvider()
```

**Step 7.5: Register in run_tests.sh**

Edit `scripts/run_tests.sh` and add:

```bash
# In the modules block:
"superdialog")
    cd superdialog && uv run pytest tests/ -v "$@"
    ;;
```

Add `superdialog` to the `list` output too.

**Step 7.6: Run the full suite**

Run: `bash scripts/run_tests.sh superdialog`
Expected: green. Triage and fix any failures that surface real bugs in the refactor.

**Step 7.7: Commit**

```bash
git add superdialog/tests/dialog_machine scripts/run_tests.sh
git commit -m "test(superdialog): port full dialog_machine test suite (500+ tests) and register runner"
```

---

## Task 8: CLI

**Files:**
- Create: `superdialog/src/superdialog/cli/__init__.py`
- Create: `superdialog/src/superdialog/cli/main.py`
- Test: `superdialog/tests/cli/test_chat.py`

**Step 8.1: Implement the CLI**

```python
# superdialog/src/superdialog/cli/main.py
import argparse
import asyncio
import json
import sys
from .. import DialogMachine, Flow, create_dialog_flow

def _cmd_chat(args):
    flow = Flow.load(args.flow)
    m = DialogMachine(flow=flow, llm=args.llm)
    async def loop():
        while True:
            try: user = input("> ")
            except (EOFError, KeyboardInterrupt): break
            if user.strip() in {"quit", "exit"}: break
            result = await m.turn(user)
            print(result.text)
    asyncio.run(loop())

def _cmd_lint(args):
    flow = Flow.load(args.flow)
    issues = []
    node_ids = {n.id for n in flow.nodes}
    for node in flow.nodes:
        for edge in node.edges or []:
            if edge.target_node_id not in node_ids:
                issues.append(f"node {node.id!r}: edge {edge.id!r} → unknown target {edge.target_node_id!r}")
    if not issues:
        print("OK"); sys.exit(0)
    for i in issues: print(i)
    sys.exit(1)

def _cmd_draw(args):
    flow = Flow.load(args.flow)
    print("graph TD")
    for node in flow.nodes:
        for edge in node.edges or []:
            print(f"  {node.id} -->|{edge.id}| {edge.target_node_id}")

def _cmd_generate(args):
    flow = asyncio.run(create_dialog_flow(prompt=args.prompt, llm=args.llm))
    json.dump(flow.model_dump(exclude_unset=True), sys.stdout, indent=2)

def main(argv=None):
    p = argparse.ArgumentParser(prog="superdialog")
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("chat"); pc.add_argument("flow"); pc.add_argument("--llm", default="openai/gpt-4o-mini"); pc.set_defaults(fn=_cmd_chat)
    pl = sub.add_parser("flow"); pls = pl.add_subparsers(dest="subcmd", required=True)
    pll = pls.add_parser("lint"); pll.add_argument("flow"); pll.set_defaults(fn=_cmd_lint)
    pld = pls.add_parser("draw"); pld.add_argument("flow"); pld.set_defaults(fn=_cmd_draw)
    plg = pls.add_parser("generate"); plg.add_argument("prompt"); plg.add_argument("--llm", default="openai/gpt-4o-mini"); plg.set_defaults(fn=_cmd_generate)
    args = p.parse_args(argv); args.fn(args)

if __name__ == "__main__":
    main()
```

**Step 8.2: Smoke test**

Run: `cd superdialog && uv run superdialog flow lint tests/fixtures/flow/kyc.json`
Expected: `OK`.

Run: `cd superdialog && uv run superdialog flow draw tests/fixtures/flow/kyc.json | head`
Expected: Mermaid `graph TD` lines.

**Step 8.3: Commit**

```bash
git add superdialog/src/superdialog/cli superdialog/tests/cli
git commit -m "feat(superdialog): CLI — chat, flow lint/draw/generate"
```

---

## Task 9: Verification sweep (Porting Skill Phase 5)

**This task must close every checklist item below. Do not skip any.**

**Step 9.1: Leftover reference scan**

```bash
cd superdialog
echo "=== from super. ===" && grep -rn "from super\." src tests
echo "=== import super. ===" && grep -rn "import super\." src tests
echo "=== super_services. ===" && grep -rn "super_services\." src tests
echo "=== langchain ===" && grep -rn "langchain" src tests
echo "=== langgraph ===" && grep -rn "langgraph" src tests
echo "=== LangGraph ===" && grep -rn "LangGraph" src tests
echo "=== livekit_flows ===" && grep -rn "livekit_flows" src tests
echo "=== SimpleFlowAgent ===" && grep -rn "SimpleFlowAgent" src tests
echo "=== livekit_bridge ===" && grep -rn "livekit_bridge" src tests
echo "=== flow_executor ===" && grep -rn "flow_executor" src tests
```

Expected: zero hits for all (modulo intentional history references in docs).

**Step 9.2: Stub scan**

```bash
cd superdialog
grep -rnE "(NotImplementedError|TODO|FIXME|^\s*pass\s*$)" src
```

Expected: zero hits in `src/`. Test fixtures may contain `pass`.

**Step 9.3: Type-check + lint**

```bash
cd superdialog
uv run pyrefly check src
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: all green.

**Step 9.4: Full test suite**

```bash
bash scripts/run_tests.sh superdialog
```

Expected: 500+ tests passing.

**Step 9.5: Smoke the spec example end-to-end**

Set `OPENAI_API_KEY`, then run the worked example from `superdialog/docs/02-api-reference.md` (KYC flow, four hosts). Confirm step 4a (CLI chat) prints sensible output for a real input.

**Step 9.6: Roadmap doc**

Update `superdialog/docs/decisions.md` §2 (Roadmap) to reflect:
- v0.1 = this port (done after Task 9)
- v0.2 = `eval/` port
- v0.3 = persistent memory backends
- v0.4 = Q4 flip to re-export shim + LiveKit voice migration
- v0.5 = decision on `langGraph/`/`langchain/`
- v1.0 = API stability + public repo split

**Step 9.7: Acceptance checklist (verbatim from Section 5 of the brainstorm)**

- [ ] `superdialog` directory has zero `from super.` / `import super.` references
- [ ] No `langchain` / `langgraph` symbols anywhere under `superdialog/src/`
- [ ] Every flow JSON in `super/` and `super_services/` round-trips through `Flow.load().save()` byte-equivalent for declared fields
- [ ] `bash scripts/run_tests.sh superdialog` green (≥ 500 tests)
- [ ] `pyrefly check superdialog/src` passes
- [ ] `ruff check superdialog/src` passes
- [ ] Spec example from `02-api-reference.md` runs end-to-end against a real OpenAI key
- [ ] `superdialog chat <flow.json>` CLI works against a real Anthropic key
- [ ] `turn_stream()` with simulated `UserChunk` stream interrupts an in-flight reply correctly in unit tests
- [ ] LiveKit plugin (`DialogMachineLLM`) verified against a local LiveKit dev server
- [ ] GAP-4 is closed (single `select_language_content` function with documented fallback policy)
- [ ] Current `super/core/voice/dialog_machine/` is untouched (`git diff super/core/voice/dialog_machine` shows zero changes)

**Step 9.8: Tag and commit**

```bash
cd superdialog
# Update version if needed
git add -A
git commit -m "chore(superdialog): v0.1.0a1 — verification sweep passes"
git tag superdialog-v0.1.0a1
```

---

## Roadmap (post-v0.1)

| Phase | Scope | Trigger |
|---|---|---|
| v0.2 | Port `eval/` (FlowEvaluator, CorpusGenerator, ResponseCache, FlowGraphAnalyzer) + `superdialog eval` CLI | OSS users ask for A/B model harness |
| v0.3 | Persistent memory backends (`RedisMemory`, `FileMemory`, `SQLiteMemory`) | Long-lived chat use case lands |
| v0.4 | **Q4 flip → A.** Make `super/core/voice/dialog_machine/__init__.py` a re-export shim; migrate `super_services` voice callers from `SimpleFlowAgent` path to `DialogMachineLLM` LiveKit plugin; delete legacy heavy adapters | v0.1 is stable in production via parallel-lives |
| v0.5 | Decision on `langGraph/`/`langchain/` (drop or port) | Only if real demand |
| v1.0 | API stability commitment, semantic versioning, split to `github.com/unpod/superdialog` | After v0.4 stabilizes |

---

## Plan completeness check

This plan covers all 10 decisions and 3 corrections from the brainstorming session. Total estimate: ~32 hours focused work, parallelizable to ~12 hours wall-clock with `agent-teams:team-feature` (Tasks 1+2 in parallel, then 3, then 4+5, then 6, then 7+8, then 9). Every code path in the engine is hard-ported with imports rewritten; every public-API symbol named in `superdialog/docs/02-api-reference.md` is implemented; no stubs survive past the task in which they're introduced.
