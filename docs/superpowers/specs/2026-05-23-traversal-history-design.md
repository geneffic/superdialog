# Traversal History — Implementation Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the traversal-history saving feature from `superdialog/chat.py` into the superdialog library so any `DialogMachine` user gets it automatically by passing `traversal_dir`.

**Architecture:** A new `superdialog/traversal/` module provides `build_traversal()` + `save_traversal()`. `ActionExecutor` records each HTTP API call into a new `action_log` on `FlowContext`. `DialogMachine` tracks chat turns internally and auto-saves on session complete.

**Tech Stack:** Python stdlib only (json, pathlib, datetime). No new dependencies.

---

## 1. New module — `superdialog/traversal/`

```
superdialog/src/superdialog/traversal/
├── __init__.py      # exports: build_traversal, save_traversal
└── traversal.py     # all logic
```

`traversal/history/` is created at runtime (gitignored). No files ship inside it.

### `traversal.py` public API

```python
def build_traversal(
    machine: DialogMachine,
    chat_turns: list[dict],    # [{bot, user, node, ts, step}, ...]
    flow: ConversationFlow,
    source: str,
    model: str,
    started_at: datetime,
) -> dict: ...

def save_traversal(traversal: dict, out_dir: Path) -> Path: ...
```

`build_traversal` is a direct port of the same function in `chat.py` with two changes:
- Reads machine state via public API (`machine.state`, `machine.is_complete`) instead of `machine._machine.context`
- Attaches API call records from `FlowContext.action_log` to each traversal step (grouped by `node_id` + `trigger`)

Output JSON schema (same keys as today, plus `actions` per step):

```json
{
  "session_id": "20260523_074400_123456",
  "flow_file": "flow_golf_ai_updated.json",
  "model": "openai/gpt-4.1-mini",
  "started_at": "...",
  "ended_at": "...",
  "is_complete": true,
  "nodes": [{"id": "...", "name": "...", "instruction": "...", "is_final": false}],
  "traversal": [
    {
      "step": 1,
      "from_node": null,
      "to_node": "greeting",
      "edge_id": null,
      "timestamp": "...",
      "node_instruction": "...",
      "bot_message": "Good evening! ...",
      "user_message": null,
      "criteria": null,
      "actions": [
        {
          "action_id": "action-auth-token",
          "trigger": "on_enter",
          "url": "https://api.example.com/auth/token",
          "method": "POST",
          "status": 200,
          "success": true,
          "result_data": {"access_token": "..."}
        }
      ]
    }
  ],
  "graph": {
    "nodes": [{"id": "...", "visited": true, "visit_count": 1, "is_final": false}],
    "edges": [{"id": "...", "source": "...", "target": "...", "condition": "...", "traversed": true, "traversed_at_step": 2}]
  }
}
```

---

## 2. `ActionRecord` model — `machine/models.py`

Add after `TransitionRecord`:

```python
class ActionRecord(BaseModel):
    action_id: str
    node_id: str
    trigger: str          # "on_enter" | "on_exit" | "edge"
    url: str
    method: str
    status: int
    success: bool
    result_data: dict
    timestamp: float = Field(default_factory=time.time)
```

Add to `FlowContext` data class:

```python
action_log: list[ActionRecord] = Field(default_factory=list)
```

Add property on `FlowContext` (parallel to `transition_log`):

```python
@property
def action_log(self) -> list[ActionRecord]:
    return self.data.action_log
```

---

## 3. `ActionExecutor` — append to `action_log`

`ActionExecutor.execute()` already receives `node_id` (the current state) and `trigger_type`. After each HTTP action completes (not skipped, not None), append one `ActionRecord` to `self._context.action_log`.

Fields to capture:
- `action_id` — from the action definition
- `node_id` — current machine state when the action runs
- `trigger` — `trigger_type.value` (e.g. `"on_enter"`)
- `url` — rendered URL (already computed in `LLMAdapter.execute_action`)
- `method` — HTTP method from action definition
- `status` — HTTP response status code
- `success` — `result.get("success", False)`
- `result_data` — `result.get("data", {})`

`LLMAdapter.execute_action` already returns the full result dict. `ActionExecutor` can read `url` + `method` from the action definition after rendering (or pass them back from `LLMAdapter`).

Record inside `LLMAdapter.execute_action` directly — it already has the rendered URL, method, status, and result data. Pass `node_id` + `trigger` into `execute_action` from `ActionExecutor`. `LLMAdapter` receives a `context: FlowContext` reference (already held as `self._context`) and appends the `ActionRecord` there.

---

## 4. `DialogMachine` — auto-save

Constructor change:

```python
DialogMachine(
    flow: Flow | FlowSet,
    llm: str,
    tools: list[Tool] | None = None,
    memory: ContextStore | None = None,
    config: dict[str, Any] | None = None,
    traversal_dir: str | Path | None = None,   # NEW
)
```

Internal tracking:

```python
self._traversal_dir: Path | None = Path(traversal_dir) if traversal_dir else None
self._chat_turns: list[dict] = []
self._session_started_at: datetime | None = None
```

`start()` — record first turn:
```python
self._session_started_at = datetime.now(timezone.utc)
self._chat_turns.append({
    "step": 1, "bot": turn.text or "", "user": None,
    "node": turn.metadata["to_node"],
    "ts": datetime.now(timezone.utc).isoformat(),
})
```

`_run_turn()` — record each turn, auto-save on complete:
```python
self._chat_turns.append({
    "step": len(self._chat_turns) + 1,
    "bot": turn.text or "", "user": text,
    "node": turn.metadata["to_node"],
    "ts": datetime.now(timezone.utc).isoformat(),
})
if self._traversal_dir and self.is_complete:
    self._auto_save_traversal()
```

`_auto_save_traversal()`:
```python
def _auto_save_traversal(self) -> None:
    from .traversal import build_traversal, save_traversal
    flow = self._flowset[self._active_flow_name]
    traversal = build_traversal(
        self, self._chat_turns, flow,
        source=self._active_flow_name,
        model=self._llm_uri,
        started_at=self._session_started_at or datetime.now(timezone.utc),
    )
    save_traversal(traversal, self._traversal_dir)
```

---

## 5. `chat.py` cleanup

Remove `build_traversal` and `save_traversal` functions. Replace with:

```python
from superdialog.traversal import build_traversal, save_traversal
```

`TRAVERSAL_DIR` constant and the save block at end of `chat()` remain — they now call the library functions. `DialogMachine` in `chat.py` does NOT set `traversal_dir` (chat.py still drives its own save to keep the CLI UI print).

---

## 6. File structure summary

| File | Change |
|------|--------|
| `superdialog/src/superdialog/traversal/__init__.py` | New — exports |
| `superdialog/src/superdialog/traversal/traversal.py` | New — ported + extended from chat.py |
| `superdialog/src/superdialog/machine/models.py` | Add `ActionRecord`, `action_log` to `FlowContext` |
| `superdialog/src/superdialog/machine/actions.py` | Append `ActionRecord` after each HTTP execution |
| `superdialog/src/superdialog/adapters/livekit.py` | Pass `node_id` + `trigger` into `execute_action` |
| `superdialog/src/superdialog/dialog_machine.py` | Add `traversal_dir` param, `_chat_turns` tracking, auto-save |
| `superdialog/chat.py` | Remove `build_traversal`/`save_traversal`, import from library |
| `superdialog/src/superdialog/traversal/history/.gitkeep` | New — keeps dir in git, contents gitignored |

---

## 7. Error handling

- `_auto_save_traversal` catches all exceptions and logs a warning — traversal failure must never crash a production turn.
- `save_traversal` creates `out_dir` if missing (`mkdir(parents=True, exist_ok=True)`).
- If `_session_started_at` is None (machine never called `start()`), fall back to `datetime.now()`.

---

## 8. Testing

- Unit test `build_traversal` with a fake `DialogMachine` stub and a minimal flow — verify JSON schema.
- Unit test `ActionRecord` appended correctly after a mock HTTP call in `ActionExecutor`.
- Integration test: run a 2-turn fake-LLM session with `traversal_dir` set, assert file written with correct steps + actions.
