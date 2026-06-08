"""Data models for the dialog state machine."""

from __future__ import annotations

import time
from typing import Any, Callable, Coroutine, Literal

from pydantic import BaseModel, Field, model_validator


class CriteriaResult(BaseModel):
    """Result of evaluating a node's completion criteria."""

    node_id: str
    criteria_met: dict[str, bool] = Field(default_factory=dict)
    all_required_met: bool = False
    user_insisting: bool = False
    recommended_edge_id: str | None = None
    reason: str = ""
    response: str | None = None
    extracted_slots: dict[str, Any] = Field(default_factory=dict)


class TurnResult(BaseModel):
    """Result of a single process_turn call — always includes a response."""

    outcome: Literal["transition", "stay", "error"]
    from_node: str
    to_node: str
    response: str
    edge_id: str | None = None
    criteria_snapshot: dict[str, bool] = Field(default_factory=dict)
    actions_fired: list[str] = Field(default_factory=list)
    error: str | None = None


class TransitionRecord(BaseModel):
    """Audit log entry for a single state transition."""

    from_node: str
    to_node: str
    edge_id: str
    criteria_met: dict[str, bool] = Field(default_factory=dict)
    skipped: bool = False
    timestamp: float = Field(default_factory=time.time)
    # Message attribution, stamped by _do_transition. user_message is the
    # caller utterance that triggered THIS transition (None for auto-routed
    # router/auto-proceed hops); bot_message is the reply generated on entry
    # to to_node. Defaults keep older persisted logs loadable.
    user_message: str | None = None
    bot_message: str = ""


class ActionRecord(BaseModel):
    """Audit log entry for a single HTTP action execution."""

    action_id: str
    node_id: str
    trigger: str  # "on_enter" | "on_exit" | "edge"
    url: str
    method: str
    status: int
    success: bool
    result_data: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class IntentFrame(BaseModel):
    """Snapshot of a suspended node for intent stack."""

    node_id: str
    slots: dict[str, Any] = Field(default_factory=dict)
    turns_spent: int = 0
    suspended_at: float = Field(default_factory=time.time)


# ToolDefinition is the single source of truth in superdialog.flow.models.
# Re-exporting it here keeps legacy callers (and the ported test suite)
# pointing at the same class FlowNode validates against — otherwise we end
# up with two distinct pydantic models named "ToolDefinition" and FlowNode
# rejects instances of the wrong one.
from superdialog.flow.models import ToolDefinition  # noqa: E402, F401


class ToolResult(BaseModel):
    """Return type from a custom tool handler.

    A plain ``dict`` returned by a handler is normalized to
    ``ToolResult(data=the_dict, transition_edge_id=None)``.
    """

    data: dict[str, Any] = Field(default_factory=dict)
    transition_edge_id: str | None = None


class ToolDescriptor(BaseModel):
    """Provider-agnostic tool representation for bridge layers."""

    id: str
    description: str
    is_data_collection: bool = False
    is_global: bool = False
    is_custom: bool = False
    input_schema: dict[str, Any] | None = None
    target_node_id: str | None = None
    handler_id: str | None = None


class ConversationData(BaseModel):
    """User-facing data: variables, history, audit trail.

    Single source of truth for all collected/computed data.
    Templates and adapters read from here.
    """

    # The ONE dict for all collected data (replaces userdata + node_slots + action_results)
    variables: dict[str, Any] = Field(default_factory=dict)

    # Conversation transcript
    history: list[dict[str, Any]] = Field(default_factory=list)

    # Criteria evaluation status per node
    criteria_status: dict[str, dict[str, bool]] = Field(default_factory=dict)

    # Audit trail
    transition_log: list[TransitionRecord] = Field(default_factory=list)
    completed_nodes: set[str] = Field(default_factory=set)
    # API action audit trail
    action_log: list[ActionRecord] = Field(default_factory=list)

    # Language/persona
    language: str = ""
    gender: str = ""

    # Rolling trace of last 3 STT-detected user languages per turn.
    # Used by the handler debounce: switch only after 2 consecutive
    # turns in a new language. Stored here for end-to-end visibility.
    language_trace: list[str] = Field(default_factory=list)

    # Source tracking for debugging "where did this value come from?"
    # Keyed by variable name → source string (e.g., "edge:greeting_to_pitch")
    sources: dict[str, str] = Field(default_factory=dict, exclude=True)

    def set_var(self, key: str, value: Any, source: str | None = None) -> None:
        """Set a variable with optional source tracking."""
        self.variables[key] = value
        if source:
            self.sources[key] = source

    def get_var(self, key: str, default: Any = None) -> Any:
        """Get a variable by key."""
        return self.variables.get(key, default)

    def merge(self, data: dict[str, Any], source: str) -> None:
        """Merge dict into variables, tracking source for each key."""
        for k, v in data.items():
            if v is not None:
                self.set_var(k, v, source=source)

    def get_source(self, key: str) -> str | None:
        """Get the source that last set a variable."""
        return self.sources.get(key)


class MachineState(BaseModel):
    """Internal machine counters — not exposed to templates or adapters."""

    current_node_id: str = ""
    visit_count: dict[str, int] = Field(default_factory=dict)
    turns_in_node: int = 0
    user_turns_in_node: int = 0
    consecutive_self_loops: int = 0
    max_self_loops: int = 2
    intent_stack: list[IntentFrame] = Field(default_factory=list)

    # Machine-tracked flag: has the executor spoken this node's content?
    # Excluded from persistent store serialization.
    node_spoken_flags: dict[str, bool] = Field(default_factory=dict, exclude=True)

    # Edge that Gate 4A denied because user hadn't spoken yet.
    # Stored so the machine can auto-retry it once the user responds,
    # without needing the LLM to call the tool again.
    pending_edge_id: str | None = Field(default=None, exclude=True)
    # collected_data that was passed with the denied edge — preserved so
    # the auto-retry can forward slot_id / player_id / etc. unchanged.
    pending_collected_data: dict[str, Any] | None = Field(default=None, exclude=True)

    @property
    def node_spoken(self) -> bool:
        """Whether the current node's content has been spoken."""
        return self.node_spoken_flags.get(self.current_node_id, False)

    @node_spoken.setter
    def node_spoken(self, value: bool) -> None:
        self.node_spoken_flags[self.current_node_id] = value

    @property
    def user_spoke_in_node(self) -> bool:
        """Whether the user has spoken at least once in this node."""
        return self.user_turns_in_node > 0


class FlowContext(BaseModel):
    """Mutable state bag that travels with the state machine.

    Combines ConversationData (user-facing) and MachineState (internal).
    Provides backward-compatible properties so existing code keeps working
    while consumers migrate to the new API.
    """

    data: ConversationData = Field(default_factory=ConversationData)
    state: MachineState = Field(default_factory=MachineState)
    session_id: str = ""

    # Runtime reference to the voice session's UserState object.
    # Set by the handler after machine creation so tools (e.g. SmsTool)
    # can access model_config, extra_data, contact_number, etc.
    user_state: Any = Field(default=None, exclude=True)

    # -- Legacy fields kept for backward compat during migration --
    node_slots: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _route_legacy_kwargs(cls, values: Any) -> Any:
        """Route old-style FlowContext kwargs to sub-models."""
        if not isinstance(values, dict):
            return values

        data_kwargs: dict[str, Any] = values.pop("data", None) or {}
        state_kwargs: dict[str, Any] = values.pop("state", None) or {}

        # Map legacy field names → ConversationData fields
        _DATA_MAP = {
            "conversation_history": "history",
            "userdata": "variables",
            "criteria_status": "criteria_status",
            "transition_log": "transition_log",
            "completed_nodes": "completed_nodes",
            "action_results": "variables",  # merged into variables
            "agent_language": "language",
            "agent_gender": "gender",
        }
        # Map legacy field names → MachineState fields
        _STATE_MAP = {
            "current_node_id": "current_node_id",
            "visit_count": "visit_count",
            "turns_in_node": "turns_in_node",
            "user_turns_in_node": "user_turns_in_node",
            "consecutive_self_loops": "consecutive_self_loops",
            "intent_stack": "intent_stack",
            "node_spoken_flags": "node_spoken_flags",
        }

        for old_key, new_key in _DATA_MAP.items():
            if old_key in values:
                val = values.pop(old_key)
                if old_key == "action_results" and new_key == "variables":
                    # Merge action_results into variables
                    existing = data_kwargs.get("variables", {})
                    existing.update(val)
                    data_kwargs["variables"] = existing
                else:
                    data_kwargs[new_key] = val

        for old_key, new_key in _STATE_MAP.items():
            if old_key in values:
                state_kwargs[new_key] = values.pop(old_key)

        if data_kwargs:
            values["data"] = (
                ConversationData(**data_kwargs)
                if isinstance(data_kwargs, dict)
                else data_kwargs
            )
        if state_kwargs:
            values["state"] = (
                MachineState(**state_kwargs)
                if isinstance(state_kwargs, dict)
                else state_kwargs
            )

        return values

    def add_message(self, role: str, content: str) -> None:
        """Append a message to conversation history."""
        self.data.history.append({"role": role, "content": content})

    def add_user_message(self, content: str) -> None:
        """Append a user message and mark that user has spoken in this node."""
        self.add_message("user", content)
        self.state.user_turns_in_node += 1
        # Mark the current node as spoken-in so the gate lets transitions
        # through. Process-turn callers used to set this implicitly via a
        # different code path; expressing it here keeps gate semantics
        # consistent across all entry points.
        self.node_spoken = True

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message to conversation history."""
        self.add_message("assistant", content)

    # ---- Backward-compatible property shims ----
    # These let existing code (machine.py, adapters, tests) work unchanged
    # while we migrate to context.data.* and context.state.*

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        return self.data.history

    @conversation_history.setter
    def conversation_history(self, value: list[dict[str, Any]]) -> None:
        self.data.history = value

    @property
    def userdata(self) -> dict[str, Any]:
        return self.data.variables

    @userdata.setter
    def userdata(self, value: dict[str, Any]) -> None:
        self.data.variables = value

    @property
    def criteria_status(self) -> dict[str, dict[str, bool]]:
        return self.data.criteria_status

    @criteria_status.setter
    def criteria_status(self, value: dict[str, dict[str, bool]]) -> None:
        self.data.criteria_status = value

    @property
    def transition_log(self) -> list[TransitionRecord]:
        return self.data.transition_log

    @transition_log.setter
    def transition_log(self, value: list[TransitionRecord]) -> None:
        self.data.transition_log = value

    @property
    def action_log(self) -> list[ActionRecord]:
        return self.data.action_log

    @action_log.setter
    def action_log(self, value: list[ActionRecord]) -> None:
        self.data.action_log = value

    @property
    def completed_nodes(self) -> set[str]:
        return self.data.completed_nodes

    @completed_nodes.setter
    def completed_nodes(self, value: set[str]) -> None:
        self.data.completed_nodes = value

    @property
    def action_results(self) -> dict[str, Any]:
        """Action results stored in variables with 'action:' source prefix."""
        return self.data.variables

    @action_results.setter
    def action_results(self, value: dict[str, Any]) -> None:
        self.data.variables.update(value)

    @property
    def agent_language(self) -> str:
        return self.data.language

    @agent_language.setter
    def agent_language(self, value: str) -> None:
        self.data.language = value

    @property
    def agent_gender(self) -> str:
        return self.data.gender

    @agent_gender.setter
    def agent_gender(self, value: str) -> None:
        self.data.gender = value

    @property
    def current_node_id(self) -> str:
        return self.state.current_node_id

    @current_node_id.setter
    def current_node_id(self, value: str) -> None:
        self.state.current_node_id = value

    @property
    def visit_count(self) -> dict[str, int]:
        return self.state.visit_count

    @visit_count.setter
    def visit_count(self, value: dict[str, int]) -> None:
        self.state.visit_count = value

    @property
    def turns_in_node(self) -> int:
        return self.state.turns_in_node

    @turns_in_node.setter
    def turns_in_node(self, value: int) -> None:
        self.state.turns_in_node = value

    @property
    def user_turns_in_node(self) -> int:
        return self.state.user_turns_in_node

    @user_turns_in_node.setter
    def user_turns_in_node(self, value: int) -> None:
        self.state.user_turns_in_node = value

    @property
    def consecutive_self_loops(self) -> int:
        return self.state.consecutive_self_loops

    @consecutive_self_loops.setter
    def consecutive_self_loops(self, value: int) -> None:
        self.state.consecutive_self_loops = value

    @property
    def MAX_SELF_LOOPS(self) -> int:  # noqa: N802
        return self.state.max_self_loops

    @property
    def intent_stack(self) -> list[IntentFrame]:
        return self.state.intent_stack

    @intent_stack.setter
    def intent_stack(self, value: list[IntentFrame]) -> None:
        self.state.intent_stack = value

    @property
    def node_spoken_flags(self) -> dict[str, bool]:
        return self.state.node_spoken_flags

    @node_spoken_flags.setter
    def node_spoken_flags(self, value: dict[str, bool]) -> None:
        self.state.node_spoken_flags = value

    @property
    def node_spoken(self) -> bool:
        return self.state.node_spoken

    @node_spoken.setter
    def node_spoken(self, value: bool) -> None:
        self.state.node_spoken = value

    @property
    def pending_edge_id(self) -> str | None:
        """Edge that Gate 4A denied — will be auto-retried when user speaks."""
        return self.state.pending_edge_id

    @pending_edge_id.setter
    def pending_edge_id(self, value: str | None) -> None:
        self.state.pending_edge_id = value

    @property
    def pending_collected_data(self) -> dict[str, Any] | None:
        """collected_data saved alongside pending_edge_id by Gate 4A."""
        return self.state.pending_collected_data

    @pending_collected_data.setter
    def pending_collected_data(self, value: dict[str, Any] | None) -> None:
        self.state.pending_collected_data = value

    @property
    def user_spoke_in_node(self) -> bool:
        return self.state.user_spoke_in_node


# Type alias for the transition callback the executor calls
TransitionRequestFn = Callable[
    [str, dict[str, Any] | None],
    Coroutine[Any, Any, "TransitionResult"],
]


class NodeScope(BaseModel):
    """Everything an executor needs to operate on a single node.

    Assembled by DialogStateMachine.build_node_scope(). The executor
    receives this and should NOT reach back into the machine for data.
    """

    # Identity
    node_id: str
    node_type: Literal["final", "static", "instruction", "router"]
    is_final: bool = False
    is_initial: bool = False
    auto_proceed: bool = False
    is_self_loop: bool = False

    # Instructions (pre-built by machine)
    system_prompt: str = ""
    node_instruction: str = ""
    speech_text: str | None = None
    language: str = ""

    # History (passed so executor's LLM has full context)
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
    )
    completed_nodes: list[str] = Field(default_factory=list)
    turns_in_node: int = 0
    visit_count: int = 1

    # Tools (descriptors — executor converts to platform-specific)
    edge_tools: list[ToolDescriptor] = Field(default_factory=list)

    # Data
    node_slots: dict[str, Any] = Field(default_factory=dict)
    userdata: dict[str, Any] = Field(default_factory=dict)

    # Criteria (for reference — machine uses these in gate checks)
    completion_criteria: list[dict[str, Any]] = Field(
        default_factory=list,
    )
    allow_skip: bool = True
    max_turns: int | None = None

    model_config = {"arbitrary_types_allowed": True}


class TransitionResult(BaseModel):
    """Result of requesting a transition through the machine gate."""

    allowed: bool
    reason: str = ""
    turn_result: TurnResult | None = None
    new_scope: NodeScope | None = None
    missing_criteria: list[str] = Field(default_factory=list)
    correction_hint: str = ""
    # Caller-facing recovery phrase spoken via TTS when the gate
    # denies a transition. Unlike correction_hint (which is for the
    # LLM), this is a natural-language phrase the caller hears so
    # the agent doesn't appear dead/hung.
    recovery_speech: str = ""
