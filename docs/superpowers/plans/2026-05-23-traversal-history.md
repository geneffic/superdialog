# Traversal History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move traversal-history saving from `superdialog/chat.py` into the library so any `DialogMachine` user gets automatic session recording by passing `traversal_dir`.

**Architecture:** Four coordinated changes — (1) `ActionRecord` model + `action_log` on `ConversationData`; (2) `LLMAdapter.execute_action` embeds rendered URL in result so `ActionExecutor` can write `ActionRecord`; (3) new `superdialog/traversal/` module with `build_traversal` + `save_traversal`; (4) `DialogMachine` tracks turns and auto-saves on complete.

**Tech Stack:** Python stdlib (json, pathlib, datetime), pydantic (already used), pytest-asyncio (already used).

---

## File map

| Action | Path |
|--------|------|
| Modify | `superdialog/src/superdialog/machine/models.py` |
| Modify | `superdialog/src/superdialog/machine/adapters/llm_adapter.py` |
| Modify | `superdialog/src/superdialog/machine/actions.py` |
| Create | `superdialog/src/superdialog/traversal/__init__.py` |
| Create | `superdialog/src/superdialog/traversal/traversal.py` |
| Modify | `superdialog/src/superdialog/dialog_machine.py` |
| Modify | `superdialog/chat.py` |
| Create | `superdialog/src/superdialog/traversal/history/.gitkeep` |
| Create | `superdialog/tests/traversal/__init__.py` |
| Create | `superdialog/tests/traversal/test_traversal.py` |

---

## Task 1: Add `ActionRecord` model and `action_log` to `ConversationData`

**Files:**
- Modify: `superdialog/src/superdialog/machine/models.py:37-50` (after `TransitionRecord`)
- Modify: `superdialog/src/superdialog/machine/models.py:90-115` (`ConversationData` fields)
- Test: `superdialog/tests/traversal/test_traversal.py`

- [ ] **Step 1: Write the failing test**

Create `superdialog/tests/traversal/__init__.py` (empty) and `superdialog/tests/traversal/test_traversal.py`:

```python
"""Tests for traversal history module."""
from __future__ import annotations

from superdialog.machine.models import ActionRecord, FlowContext


def test_action_record_fields():
    rec = ActionRecord(
        action_id="action-auth-token",
        node_id="greeting",
        trigger="on_enter",
        url="https://api.example.com/auth/token",
        method="POST",
        status=200,
        success=True,
        result_data={"access_token": "abc123"},
    )
    assert rec.action_id == "action-auth-token"
    assert rec.trigger == "on_enter"
    assert rec.result_data == {"access_token": "abc123"}
    assert rec.timestamp > 0


def test_flow_context_action_log_starts_empty():
    ctx = FlowContext()
    assert ctx.action_log == []


def test_flow_context_action_log_append():
    ctx = FlowContext()
    rec = ActionRecord(
        action_id="action-players-search",
        node_id="greeting",
        trigger="on_enter",
        url="https://api.example.com/players/search",
        method="GET",
        status=400,
        success=False,
        result_data={"detail": {"error": "invalid_request"}},
    )
    ctx.action_log.append(rec)
    assert len(ctx.action_log) == 1
    assert ctx.action_log[0].action_id == "action-players-search"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd superdialog && python -m pytest tests/traversal/test_traversal.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'ActionRecord'`

- [ ] **Step 3: Add `ActionRecord` to `models.py` after `TransitionRecord` (line 46)**

In `superdialog/src/superdialog/machine/models.py`, insert after the `TransitionRecord` class (after line 46, before `class IntentFrame`):

```python
class ActionRecord(BaseModel):
    """Audit log entry for a single HTTP action execution."""

    action_id: str
    node_id: str
    trigger: str          # "on_enter" | "on_exit" | "edge"
    url: str
    method: str
    status: int
    success: bool
    result_data: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
```

- [ ] **Step 4: Add `action_log` field to `ConversationData`**

In `ConversationData` (around line 100, in the "Audit trail" comment block), add after `completed_nodes`:

```python
    # API action audit trail
    action_log: list[ActionRecord] = Field(default_factory=list)
```

- [ ] **Step 5: Add `action_log` property shim to `FlowContext`**

In `FlowContext`, after the `transition_log` property shim (around line 310), add:

```python
    @property
    def action_log(self) -> list[ActionRecord]:
        return self.data.action_log

    @action_log.setter
    def action_log(self, value: list[ActionRecord]) -> None:
        self.data.action_log = value
```

- [ ] **Step 6: Run test to verify it passes**

```bash
cd superdialog && python -m pytest tests/traversal/test_traversal.py -v 2>&1 | head -20
```

Expected: `3 passed`

- [ ] **Step 7: Run full suite to check no regressions**

```bash
cd superdialog && python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: same pass/skip counts as before (≥359 passed)

---

## Task 2: Record `ActionRecord` in `LLMAdapter` + `ActionExecutor`

**Files:**
- Modify: `superdialog/src/superdialog/machine/adapters/llm_adapter.py:259-280`
- Modify: `superdialog/src/superdialog/machine/actions.py:60-80`
- Test: `superdialog/tests/traversal/test_traversal.py`

- [ ] **Step 1: Write the failing test**

Add to `superdialog/tests/traversal/test_traversal.py`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock
from superdialog.machine.actions import ActionExecutor
from superdialog.machine.models import FlowContext, ActionRecord
from superdialog.flow.models import ActionTriggerType


def _make_action_trigger(action_id: str):
    """Build a minimal action trigger matching what ActionExecutor expects."""
    from superdialog.flow.models import NodeAction
    trigger = MagicMock()
    trigger.action_id = action_id
    trigger.trigger_type = ActionTriggerType.ON_ENTER
    return trigger


def _make_custom_action(action_id: str):
    from superdialog.flow.models import CustomAction, HttpMethod
    action = MagicMock()
    action.id = action_id
    action.store_response_as = "auth_result"
    action.env_updates = []
    action.run_once = False
    return action


def test_action_executor_records_action_log():
    ctx = FlowContext()
    ctx.state.current_node_id = "greeting"

    custom_action = _make_custom_action("action-auth-token")
    action_map = {"action-auth-token": custom_action}

    adapter = MagicMock()
    adapter.execute_action = AsyncMock(return_value={
        "status": 200,
        "success": True,
        "data": {"access_token": "tok123"},
        "_rendered_url": "https://api.example.com/auth/token",
        "_method": "POST",
    })

    executor = ActionExecutor(adapter=adapter, action_map=action_map)
    trigger = _make_action_trigger("action-auth-token")

    asyncio.run(executor.execute([trigger], ctx, trigger_type=ActionTriggerType.ON_ENTER))

    assert len(ctx.action_log) == 1
    rec = ctx.action_log[0]
    assert rec.action_id == "action-auth-token"
    assert rec.node_id == "greeting"
    assert rec.trigger == "on_enter"
    assert rec.url == "https://api.example.com/auth/token"
    assert rec.method == "POST"
    assert rec.status == 200
    assert rec.success is True
    assert rec.result_data == {"access_token": "tok123"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd superdialog && python -m pytest tests/traversal/test_traversal.py::test_action_executor_records_action_log -v 2>&1 | head -20
```

Expected: FAIL — `ctx.action_log` is empty (ActionExecutor doesn't record yet)

- [ ] **Step 3: Add `_rendered_url` and `_method` to `execute_action` result**

In `superdialog/src/superdialog/machine/adapters/llm_adapter.py`, in `execute_action`, after the `result` dict is built (around line 266, where `result["data"] = response.json()` is set), add:

```python
                result["_rendered_url"] = url
                result["_method"] = method
```

Also add the same two keys to the `error_result` at the bottom of `execute_action`:

```python
        except Exception as exc:
            print(f"[TRACK] LLMAdapter - action={action.id} HTTP error: {exc}")
            error_result: dict[str, Any] = {
                "success": False,
                "error": str(exc),
                "status": 0,
                "_rendered_url": url,
                "_method": method,
            }
            return error_result
```

- [ ] **Step 4: Record `ActionRecord` in `ActionExecutor.execute` after getting result**

In `superdialog/src/superdialog/machine/actions.py`, after the block that stores `result` in `context.userdata` (after the `env_updates` loop, around line 100), add:

```python
                # Record API call in action_log for traversal history
                if result is not None:
                    from superdialog.machine.models import ActionRecord
                    rendered_url = result.pop("_rendered_url", "")
                    method_str = result.pop("_method", "")
                    context.action_log.append(ActionRecord(
                        action_id=action.id,
                        node_id=context.current_node_id,
                        trigger=trigger_type.value if trigger_type is not None else "unknown",
                        url=rendered_url,
                        method=method_str,
                        status=result.get("status", 0),
                        success=bool(result.get("success", False)),
                        result_data=result.get("data", {}),
                    ))
```

Important: the `result.pop("_rendered_url", "")` and `result.pop("_method", "")` calls must happen **before** `context.userdata[action.store_response_as] = result` so the internal keys don't leak into userdata. Move these pop calls to just after `result = await self._adapter.execute_action(...)` returns, before any storage logic.

The correct placement in `actions.py` (replace the existing block from `result = await self._adapter.execute_action(...)` through the `env_updates` loop):

```python
                result = await self._adapter.execute_action(action, context.userdata)
                fired.append(action.id)

                # Strip internal metadata keys before storing
                rendered_url = ""
                method_str = ""
                if isinstance(result, dict):
                    rendered_url = result.pop("_rendered_url", "")
                    method_str = result.pop("_method", "")

                print(f"[TRACK] ActionExecutor - action {action.id} result: {result}")
                if result is not None and action.store_response_as:
                    print(
                        f"[TRACK] ActionExecutor - storing result as: {action.store_response_as}, result: {result}"
                    )
                    context.userdata[action.store_response_as] = result
                    context.data.merge(
                        {action.store_response_as: result},
                        source=f"action:{action.id}",
                    )
                    print(
                        f"[TRACK] ActionExecutor - AFTER STORE - userdata keys: {list(context.userdata.keys())}"
                    )
                if result is not None and action.env_updates:
                    for update in action.env_updates:
                        try:
                            value = result
                            for key in update.result_path.split("."):
                                value = value[key]
                            context.userdata[update.env_key] = str(value)
                            logger.info(
                                "[ActionExecutor] env_update applied: %s = <token> (from %s)",
                                update.env_key,
                                update.result_path,
                            )
                        except (KeyError, TypeError, IndexError):
                            logger.warning(
                                "[ActionExecutor] env_update: could not resolve '%s' "
                                "from action '%s' result",
                                update.result_path,
                                action.id,
                            )

                # Record API call in action_log for traversal history
                if result is not None:
                    from superdialog.machine.models import ActionRecord
                    context.action_log.append(ActionRecord(
                        action_id=action.id,
                        node_id=context.current_node_id,
                        trigger=trigger_type.value if trigger_type is not None else "unknown",
                        url=rendered_url,
                        method=method_str,
                        status=result.get("status", 0),
                        success=bool(result.get("success", False)),
                        result_data=result.get("data", {}) if isinstance(result.get("data"), dict) else {},
                    ))
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd superdialog && python -m pytest tests/traversal/test_traversal.py -v 2>&1 | head -20
```

Expected: `4 passed`

- [ ] **Step 6: Run full suite to check no regressions**

```bash
cd superdialog && python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: same pass/skip counts as before

---

## Task 3: Create `superdialog/traversal/` module

**Files:**
- Create: `superdialog/src/superdialog/traversal/__init__.py`
- Create: `superdialog/src/superdialog/traversal/traversal.py`
- Create: `superdialog/src/superdialog/traversal/history/.gitkeep`
- Test: `superdialog/tests/traversal/test_traversal.py`

- [ ] **Step 1: Write the failing test**

Add to `superdialog/tests/traversal/test_traversal.py`:

```python
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from superdialog.traversal import build_traversal, save_traversal


def _make_fake_machine(nodes, transition_log=None, action_log=None, is_complete=True):
    """Build a minimal fake DialogMachine for build_traversal."""
    machine = MagicMock()
    machine.is_complete = is_complete
    machine.state = {"node_id": nodes[-1]["id"] if nodes else "", "slots": {}}

    ctx = MagicMock()
    ctx.transition_log = transition_log or []
    ctx.visit_count = {n["id"]: 1 for n in nodes}
    ctx.action_log = action_log or []

    inner = MagicMock()
    inner.context = ctx
    machine._machine = inner

    return machine


def _make_flow(node_ids):
    """Build a minimal fake ConversationFlow."""
    flow = MagicMock()
    flow_nodes = []
    for nid in node_ids:
        n = MagicMock()
        n.id = nid
        n.name = nid.replace("_", " ").title()
        n.instruction = f"Instruction for {nid}"
        n.static_text = None
        n.is_final = (nid == node_ids[-1])
        n.edges = []
        flow_nodes.append(n)
    flow.nodes = flow_nodes
    flow.global_edges = []
    return flow


def test_build_traversal_schema():
    from superdialog.machine.models import TransitionRecord

    nodes = [{"id": "greet"}, {"id": "collect_name"}, {"id": "done"}]
    flow = _make_flow(["greet", "collect_name", "done"])

    tr = TransitionRecord(
        from_node="greet",
        to_node="collect_name",
        edge_id="greet_to_collect",
        criteria_met={"name_asked": True},
        skipped=False,
        timestamp=1700000001.0,
    )

    machine = _make_fake_machine(nodes, transition_log=[tr], is_complete=True)
    chat_turns = [
        {"step": 1, "bot": "Hello!", "user": None, "node": "greet", "ts": "2026-05-23T00:00:00Z"},
        {"step": 2, "bot": "What is your name?", "user": "Ankit", "node": "collect_name", "ts": "2026-05-23T00:00:05Z"},
    ]
    started_at = datetime(2026, 5, 23, 0, 0, 0, tzinfo=timezone.utc)

    result = build_traversal(machine, chat_turns, flow, "kyc.json", "openai/gpt-4.1-mini", started_at)

    assert result["flow_file"] == "kyc.json"
    assert result["model"] == "openai/gpt-4.1-mini"
    assert result["is_complete"] is True
    assert len(result["traversal"]) == 2
    step1 = result["traversal"][0]
    assert step1["step"] == 1
    assert step1["bot_message"] == "Hello!"
    assert step1["user_message"] is None
    assert step1["from_node"] is None
    step2 = result["traversal"][1]
    assert step2["step"] == 2
    assert step2["from_node"] == "greet"
    assert step2["to_node"] == "collect_name"
    assert step2["edge_id"] == "greet_to_collect"
    assert step2["user_message"] == "Ankit"
    assert "actions" in step2

    assert "graph" in result
    graph_node_ids = [n["id"] for n in result["graph"]["nodes"]]
    assert "greet" in graph_node_ids
    assert result["graph"]["edges"][0]["traversed"] is True


def test_save_traversal_creates_file():
    traversal = {
        "session_id": "20260523_000000_000000",
        "flow_file": "test.json",
        "model": "test",
        "started_at": "2026-05-23T00:00:00+00:00",
        "ended_at": "2026-05-23T00:01:00+00:00",
        "is_complete": True,
        "nodes": [],
        "traversal": [],
        "graph": {"nodes": [], "edges": []},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_traversal(traversal, Path(tmpdir))
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["session_id"] == "20260523_000000_000000"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd superdialog && python -m pytest tests/traversal/test_traversal.py::test_build_traversal_schema tests/traversal/test_traversal.py::test_save_traversal_creates_file -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'build_traversal'`

- [ ] **Step 3: Create `superdialog/src/superdialog/traversal/history/.gitkeep`**

```bash
mkdir -p superdialog/src/superdialog/traversal/history
touch superdialog/src/superdialog/traversal/history/.gitkeep
```

Add to `.gitignore` (or create `superdialog/src/superdialog/traversal/history/.gitignore`):

```
# Ignore all traversal files but keep the directory
*
!.gitkeep
!.gitignore
```

- [ ] **Step 4: Create `superdialog/src/superdialog/traversal/traversal.py`**

This is a port of `build_traversal` + `save_traversal` from `chat.py` with two changes:
- Reads machine state via `machine._machine.context` (library-internal, acceptable)
- Attaches `actions` per traversal step from `context.action_log`

```python
"""Traversal history builder — records a full dialog session to JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superdialog.dialog_machine import DialogMachine
    from superdialog.flow.models import ConversationFlow


def build_traversal(
    machine: "DialogMachine",
    chat_turns: list[dict[str, Any]],
    flow: "ConversationFlow",
    source: str,
    model: str,
    started_at: datetime,
) -> dict[str, Any]:
    """Build traversal JSON from a completed chat session.

    Args:
        machine: The DialogMachine instance after the session.
        chat_turns: List of dicts with keys: step, bot, user, node, ts.
        flow: The ConversationFlow used in the session.
        source: Display name for the flow file (e.g. "flow.json").
        model: Model URI used (e.g. "openai/gpt-4.1-mini").
        started_at: UTC datetime when the session started.
    """
    ended_at = datetime.now(timezone.utc)
    session_id = started_at.strftime("%Y%m%d_%H%M%S_%f")[:20]

    # Node lookup: id -> {id, name, instruction, is_final}
    node_lookup: dict[str, dict[str, Any]] = {
        n.id: {
            "id": n.id,
            "name": n.name,
            "instruction": n.instruction or (n.static_text if hasattr(n, "static_text") else "") or "",
            "is_final": n.is_final,
        }
        for n in flow.nodes
    }

    # All edges from flow (node edges + global edges)
    all_edges: list[dict[str, Any]] = []
    for n in flow.nodes:
        for e in n.edges:
            all_edges.append({
                "id": e.id,
                "source": n.id,
                "target": e.target_node_id,
                "condition": e.condition or "",
            })
    for e in getattr(flow, "global_edges", []):
        all_edges.append({
            "id": e.id,
            "source": "__global__",
            "target": e.target_node_id,
            "condition": e.condition or "",
        })

    # Pull state from machine internals
    transition_log: list[Any] = []
    visit_count: dict[str, int] = {}
    action_log: list[Any] = []
    is_complete = False
    if machine._machine is not None:
        ctx = machine._machine.context
        transition_log = list(ctx.transition_log)
        visit_count = dict(ctx.visit_count)
        action_log = list(ctx.action_log)
        is_complete = machine._machine.is_complete

    # Build action lookup: node_id -> list of ActionRecord dicts
    actions_by_node: dict[str, list[dict[str, Any]]] = {}
    for rec in action_log:
        entry = {
            "action_id": rec.action_id,
            "trigger": rec.trigger,
            "url": rec.url,
            "method": rec.method,
            "status": rec.status,
            "success": rec.success,
            "result_data": rec.result_data,
        }
        actions_by_node.setdefault(rec.node_id, []).append(entry)

    # Build traversal steps
    traversal_steps: list[dict[str, Any]] = []

    if chat_turns:
        first_turn = chat_turns[0]
        first_node = first_turn.get("node", "")
        traversal_steps.append({
            "step": 1,
            "from_node": None,
            "to_node": first_node,
            "edge_id": None,
            "timestamp": first_turn.get("ts", ""),
            "node_instruction": node_lookup.get(first_node, {}).get("instruction", ""),
            "bot_message": first_turn.get("bot", ""),
            "user_message": None,
            "criteria": None,
            "actions": actions_by_node.get(first_node, []),
        })

    for i, rec in enumerate(transition_log):
        turn = chat_turns[i + 1] if i + 1 < len(chat_turns) else {}
        criteria = {
            "met": bool(rec.criteria_met) and all(rec.criteria_met.values()) and not rec.skipped,
            "skipped": rec.skipped,
            "edge_id": rec.edge_id,
            "criteria_map": dict(rec.criteria_met),
        }
        traversal_steps.append({
            "step": i + 2,
            "from_node": rec.from_node,
            "to_node": rec.to_node,
            "edge_id": rec.edge_id,
            "timestamp": datetime.fromtimestamp(rec.timestamp, tz=timezone.utc).isoformat(),
            "node_instruction": node_lookup.get(rec.to_node, {}).get("instruction", ""),
            "bot_message": turn.get("bot"),
            "user_message": turn.get("user"),
            "criteria": criteria,
            "actions": actions_by_node.get(rec.to_node, []),
        })

    # Graph annotations
    traversed_edge_ids = {rec.edge_id for rec in transition_log}
    edge_step_map = {rec.edge_id: (i + 2) for i, rec in enumerate(transition_log)}

    graph_nodes = [
        {
            "id": n.id,
            "name": n.name,
            "visited": n.id in visit_count,
            "visit_count": visit_count.get(n.id, 0),
            "is_final": n.is_final,
        }
        for n in flow.nodes
    ]
    graph_edges = [
        {
            "id": e["id"],
            "source": e["source"],
            "target": e["target"],
            "condition": e["condition"],
            "traversed": e["id"] in traversed_edge_ids,
            "traversed_at_step": edge_step_map.get(e["id"]),
        }
        for e in all_edges
    ]

    return {
        "session_id": session_id,
        "flow_file": source,
        "model": model,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "is_complete": is_complete,
        "nodes": list(node_lookup.values()),
        "traversal": traversal_steps,
        "graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
        },
    }


def save_traversal(traversal: dict[str, Any], out_dir: Path) -> Path:
    """Write traversal JSON to out_dir. Creates dir if missing. Returns path written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = traversal["session_id"]
    path = out_dir / f"traversal_{session_id}.json"
    path.write_text(json.dumps(traversal, indent=2, ensure_ascii=False))
    return path
```

- [ ] **Step 5: Create `superdialog/src/superdialog/traversal/__init__.py`**

```python
"""Traversal history — build and save dialog session recordings."""

from .traversal import build_traversal, save_traversal

__all__ = ["build_traversal", "save_traversal"]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd superdialog && python -m pytest tests/traversal/ -v 2>&1 | head -30
```

Expected: all traversal tests pass

- [ ] **Step 7: Run full suite**

```bash
cd superdialog && python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: same pass/skip counts as before

---

## Task 4: `DialogMachine` — `traversal_dir`, turn tracking, auto-save

**Files:**
- Modify: `superdialog/src/superdialog/dialog_machine.py`
- Test: `superdialog/tests/traversal/test_traversal.py`

- [ ] **Step 1: Write the failing test**

Add to `superdialog/tests/traversal/test_traversal.py`:

```python
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from superdialog import DialogMachine, Flow

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "flow"


class _ScriptedProvider:
    def __init__(self, responses):
        self._responses = list(responses)

    async def complete(self, messages, tools=None, **opts):
        from superdialog.llm.provider import CompletionResult
        text = self._responses.pop(0) if self._responses else "{}"
        return CompletionResult(text=text, tool_calls=[], metadata={})

    async def stream(self, messages, tools=None, **opts):
        result = await self.complete(messages)
        from superdialog.llm.provider import StreamChunk as SC
        yield SC(text=result.text, tool_call_delta=None, done=True)


def _criteria(edge, response="ok", all_met=True):
    return json.dumps({
        "criteria_met": {},
        "extracted_slots": {},
        "all_required_met": all_met,
        "user_insisting": False,
        "recommended_edge_id": edge,
        "reason": "test",
        "response": response,
    })


@pytest.mark.asyncio
async def test_dialog_machine_auto_saves_traversal():
    flow = Flow.load(FIXTURE_DIR / "kyc.json")

    responses = [
        "Hello, please provide your Aadhaar last 4.",          # greet static/start
        _criteria("greet_to_name", "What's your name?"),       # greet eval
        _criteria("name_to_dob", "Date of birth?"),            # collect_name eval
        _criteria("dob_to_pan", "PAN number?"),                # collect_dob eval
        _criteria("pan_to_done", "Thank you!", all_met=True),  # collect_pan eval
    ]
    provider = _ScriptedProvider(responses)

    with tempfile.TemporaryDirectory() as tmpdir:
        dm = DialogMachine(
            flow=flow,
            llm="openai/gpt-4.1-mini",
            traversal_dir=tmpdir,
        )
        dm._llm = provider

        await dm.start()
        await dm.turn("Ankit")
        await dm.turn("01/01/1990")
        await dm.turn("ABCDE1234F")

        files = list(Path(tmpdir).glob("traversal_*.json"))
        assert len(files) == 1, f"Expected 1 traversal file, got {files}"

        data = json.loads(files[0].read_text())
        assert data["is_complete"] is True
        assert len(data["traversal"]) >= 1
        assert data["model"] == "openai/gpt-4.1-mini"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd superdialog && python -m pytest tests/traversal/test_traversal.py::test_dialog_machine_auto_saves_traversal -v 2>&1 | head -20
```

Expected: FAIL — `DialogMachine.__init__` doesn't accept `traversal_dir`

- [ ] **Step 3: Add imports and `traversal_dir` param to `DialogMachine.__init__`**

In `superdialog/src/superdialog/dialog_machine.py`, add to imports at top:

```python
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal
```

Change `__init__` signature and body:

```python
    def __init__(
        self,
        flow: Flow | FlowSet,
        llm: str,
        tools: list[Tool] | None = None,
        memory: ContextStore | None = None,
        config: dict[str, Any] | None = None,
        traversal_dir: str | Path | None = None,
    ) -> None:
        self._flowset: FlowSet = (
            flow if isinstance(flow, FlowSet) else FlowSet({"main": flow})
        )
        self._active_flow_name = next(iter(self._flowset.names()))
        self._llm_uri = llm
        self._llm: LLMProvider = resolve_llm(llm)
        self._tools = list(tools or [])
        self._memory: ContextStore = memory or InMemoryContextStore()
        self._config: dict[str, Any] = dict(config or {})
        self._pending_system_messages: list[str] = list(
            self._config.pop(_SYSTEM_MARKER_KEY, [])
        )
        self._machine: DialogStateMachine | None = None
        self._adapter: LLMAdapter | None = None
        self._pending_chat_ctx: Any = None
        self._pending_flow_state: Any = None
        # Traversal tracking
        self._traversal_dir: Path | None = Path(traversal_dir) if traversal_dir else None
        self._chat_turns: list[dict[str, Any]] = []
        self._session_started_at: datetime | None = None
```

- [ ] **Step 4: Record first turn in `start()`**

In `start()`, after `return Turn(...)`, add the turn-recording before the return:

```python
        # Record initial turn for traversal
        self._session_started_at = datetime.now(timezone.utc)
        self._chat_turns = [{
            "step": 1,
            "bot": response or "",
            "user": None,
            "node": current.id,
            "ts": datetime.now(timezone.utc).isoformat(),
        }]
        return Turn(
            text=response,
            ...
        )
```

- [ ] **Step 5: Record each turn and auto-save in `_run_turn()`**

In `_run_turn()`, after `result = await machine.process_turn(text)` and before the `return Turn(...)`, add:

```python
        # Record turn for traversal history
        self._chat_turns.append({
            "step": len(self._chat_turns) + 1,
            "bot": result.response or "",
            "user": text,
            "node": result.to_node,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        # Auto-save traversal when session completes
        if self._traversal_dir and machine.is_complete:
            self._auto_save_traversal()
```

- [ ] **Step 6: Add `_auto_save_traversal()` method**

Add as a private method in `DialogMachine`, after `_consume_pending_system_messages`:

```python
    def _auto_save_traversal(self) -> None:
        """Save traversal JSON to _traversal_dir. Swallows all errors."""
        try:
            from .traversal import build_traversal, save_traversal
            flow = self._flowset[self._active_flow_name]
            traversal = build_traversal(
                self,
                self._chat_turns,
                flow,
                source=self._active_flow_name,
                model=self._llm_uri,
                started_at=self._session_started_at or datetime.now(timezone.utc),
            )
            path = save_traversal(traversal, self._traversal_dir)
            logger.info("[DialogMachine] traversal saved: %s", path)
        except Exception:
            logger.warning("[DialogMachine] traversal save failed", exc_info=True)
```

- [ ] **Step 7: Run test to verify it passes**

```bash
cd superdialog && python -m pytest tests/traversal/test_traversal.py::test_dialog_machine_auto_saves_traversal -v 2>&1 | head -20
```

Expected: PASS

- [ ] **Step 8: Run full suite**

```bash
cd superdialog && python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: same pass/skip counts as before

---

## Task 5: Update `chat.py` to use library functions

**Files:**
- Modify: `superdialog/chat.py`

- [ ] **Step 1: Replace `build_traversal` and `save_traversal` in `chat.py`**

In `superdialog/chat.py`:

1. Remove the entire `build_traversal` function (lines ~120–215).
2. Remove the entire `save_traversal` function (lines ~218–225).
3. Add import at the top (after existing imports):

```python
from superdialog.traversal import build_traversal, save_traversal
```

The `TRAVERSAL_DIR`, `chat_turns` tracking, and the save block at the end of `chat()` remain unchanged — `chat.py` still drives its own save so it can print the CLI confirmation message.

- [ ] **Step 2: Verify `chat.py` still runs**

```bash
cd superdialog && python chat.py --help
```

Expected: prints help text without import errors.

- [ ] **Step 3: Run full suite one final time**

```bash
cd superdialog && python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: same pass/skip counts as before, no regressions.

---

## Self-review

**Spec coverage check:**

| Spec requirement | Covered by task |
|-----------------|----------------|
| `superdialog/traversal/traversal.py` (renamed from recorder.py) | Task 3 |
| `ActionRecord` model | Task 1 |
| `action_log` on `FlowContext` | Task 1 |
| Record URL + result in `ActionExecutor` | Task 2 |
| `DialogMachine(traversal_dir=...)` | Task 4 |
| Track `_chat_turns` in `start()` + `_run_turn()` | Task 4 |
| Auto-save on `is_complete` | Task 4 |
| `chat.py` cleanup | Task 5 |
| `traversal/history/.gitkeep` | Task 3 |
| Error handling (swallow traversal errors) | Task 4 step 6 |

**Placeholder scan:** No TBDs, no vague steps.

**Type consistency:** `build_traversal` signature consistent across Task 3 (definition), Task 4 (call in `_auto_save_traversal`), and Task 3 test. `ActionRecord` fields consistent between Task 1 (model), Task 2 (append), Task 3 (access in `build_traversal`).
