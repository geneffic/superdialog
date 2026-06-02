"""SDK entry point for running flows end-to-end."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from superdialog.flow.models import ConversationFlow
from superdialog.machine.adapters.text_adapter import TextAdapter
from superdialog.machine.criteria import CriteriaJudge, LLMCallable
from superdialog.machine.machine import DialogStateMachine
from superdialog.machine.models import TransitionRecord

logger = logging.getLogger(__name__)


class FlowResult(BaseModel):
    """Result of running a flow end-to-end."""

    final_state: str
    is_complete: bool
    transitions: list[TransitionRecord] = Field(default_factory=list)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    responses: list[str] = Field(default_factory=list)
    userdata: dict[str, Any] = Field(default_factory=dict)


def _load_flow(flow: ConversationFlow | str) -> ConversationFlow:
    """Load flow from ConversationFlow, JSON string, or file path."""
    if isinstance(flow, ConversationFlow):
        return flow
    if isinstance(flow, str):
        stripped = flow.strip()
        if stripped.startswith("{"):
            return ConversationFlow.from_json_string(flow)
        path = Path(flow)
        if path.exists() and path.is_file():
            return ConversationFlow.from_file(path)
        return ConversationFlow.from_json_string(flow)
    msg = f"Expected ConversationFlow, str, or file path, got {type(flow)}"
    raise TypeError(msg)


async def create_machine(
    flow: ConversationFlow | str,
    llm_fn: LLMCallable,
    system_prompt: str = "",
    session_id: str | None = None,
    store: Any | None = None,
) -> DialogStateMachine:
    """Create a DialogStateMachine wired with TextAdapter + CriteriaJudge.

    Use this for interactive per-turn usage where you feed user input
    one message at a time via ``machine.process_turn(user_input)``.

    Args:
        flow: A ConversationFlow, JSON string, or path to a JSON file.
        llm_fn: Async callable ``(messages) -> str`` for LLM calls.
        system_prompt: Optional override for the flow's system_prompt.
        session_id: Optional session ID for context persistence.
        store: Optional ContextStore for saving/loading context.

    Returns:
        A ready-to-use DialogStateMachine instance.
    """
    logger.info("[RUNNER] create_machine starting with session_id=%s", session_id)
    loaded = _load_flow(flow)
    logger.info(
        "[RUNNER] flow loaded: nodes=%d, edges=%d, initial_node=%s",
        len(loaded.nodes),
        sum(len(n.edges) for n in loaded.nodes),
        loaded.initial_node,
    )
    prompt = system_prompt or loaded.system_prompt

    logger.info("[RUNNER] creating CriteriaJudge and TextAdapter")
    judge = CriteriaJudge(llm_fn=llm_fn)
    adapter = TextAdapter(
        llm_fn=llm_fn,
        criteria_judge=judge,
        system_prompt=prompt,
    )

    logger.info("[RUNNER] creating DialogStateMachine from flow")
    machine = await DialogStateMachine.from_flow(
        loaded, adapter, session_id=session_id, store=store
    )
    logger.info(
        "[RUNNER] machine created successfully, current_state=%s, is_complete=%s",
        machine.current_state,
        machine.is_complete,
    )
    return machine


async def run_flow(
    flow: ConversationFlow | str,
    llm_fn: LLMCallable,
    user_messages: list[str],
    system_prompt: str = "",
) -> FlowResult:
    """Run a flow end-to-end with scripted user messages.

    Args:
        flow: A ConversationFlow, JSON string, or path to a JSON file.
        llm_fn: Async callable ``(messages) -> str`` for LLM calls.
        user_messages: Scripted user inputs fed one per turn.
        system_prompt: Optional override for the flow's system_prompt.

    Returns:
        FlowResult with final state, transitions, and conversation log.
    """
    logger.info("[RUNNER] run_flow starting with %d user_messages", len(user_messages))
    loaded = _load_flow(flow)
    logger.info(
        "[RUNNER] flow loaded for run_flow: nodes=%d, initial_node=%s",
        len(loaded.nodes),
        loaded.initial_node,
    )
    prompt = system_prompt or loaded.system_prompt

    logger.info("[RUNNER] creating CriteriaJudge and TextAdapter for run_flow")
    judge = CriteriaJudge(llm_fn=llm_fn)
    adapter = TextAdapter(
        llm_fn=llm_fn,
        criteria_judge=judge,
        system_prompt=prompt,
    )

    logger.info("[RUNNER] creating machine for run_flow")
    machine = await DialogStateMachine.from_flow(loaded, adapter)

    responses: list[str] = []
    for i, msg in enumerate(user_messages, 1):
        if machine.is_complete:
            logger.info(
                "[RUNNER] flow completed early at turn %d, state=%s",
                i,
                machine.current_state,
            )
            break
        logger.info(
            "[RUNNER] processing turn %d/%d: %s",
            i,
            len(user_messages),
            msg[:50] + "..." if len(msg) > 50 else msg,
        )
        result = await machine.process_turn(msg)
        if result.response:
            logger.info(
                "[RUNNER] turn %d response received, length=%d", i, len(result.response)
            )
            responses.append(result.response)
        else:
            logger.info("[RUNNER] turn %d: no response generated", i)

    logger.info(
        "[RUNNER] run_flow completed: final_state=%s, is_complete=%s, responses=%d",
        machine.current_state,
        machine.is_complete,
        len(responses),
    )
    return FlowResult(
        final_state=machine.current_state,
        is_complete=machine.is_complete,
        transitions=list(machine.context.data.transition_log),
        conversation_history=list(machine.context.data.history),
        responses=responses,
        userdata=dict(machine.context.data.variables),
    )


async def run_flow_from_node(
    flow: ConversationFlow | str,
    llm_fn: LLMCallable,
    start_node: str,
    user_messages: list[str],
    system_prompt: str = "",
) -> FlowResult:
    """Run a flow starting from a specific node (direct state injection).

    Positions the machine at ``start_node`` by directly setting
    pytransitions state — no fast-forward messages needed.

    Args:
        flow: A ConversationFlow, JSON string, or path to a JSON file.
        llm_fn: Async callable ``(messages) -> str`` for LLM calls.
        start_node: Node ID to start execution from.
        user_messages: Scripted user inputs fed one per turn.
        system_prompt: Optional override for the flow's system_prompt.

    Returns:
        FlowResult with final state, transitions, and conversation log.

    Raises:
        ValueError: If start_node is not a valid node in the flow.
    """
    loaded = _load_flow(flow)
    node_ids = {n.id for n in loaded.nodes}
    if start_node not in node_ids:
        msg = f"Node '{start_node}' not found in flow"
        raise ValueError(msg)

    prompt = system_prompt or loaded.system_prompt

    judge = CriteriaJudge(llm_fn=llm_fn)
    adapter = TextAdapter(
        llm_fn=llm_fn,
        criteria_judge=judge,
        system_prompt=prompt,
    )

    machine = await DialogStateMachine.from_flow(loaded, adapter)

    # Direct state injection — skip to target node
    if start_node != loaded.initial_node:
        machine.state = start_node
        machine.context.state.current_node_id = start_node
        machine.context.state.visit_count[start_node] = (
            machine.context.state.visit_count.get(start_node, 0) + 1
        )

    responses: list[str] = []
    for msg in user_messages:
        if machine.is_complete:
            break
        result = await machine.process_turn(msg)
        if result.response:
            responses.append(result.response)

    return FlowResult(
        final_state=machine.current_state,
        is_complete=machine.is_complete,
        transitions=list(machine.context.data.transition_log),
        conversation_history=list(machine.context.data.history),
        responses=responses,
        userdata=dict(machine.context.data.variables),
    )


async def run_flow_toolcall(
    flow: ConversationFlow | str,
    model_id: str,
    user_messages: list[str],
    system_prompt: str = "",
) -> FlowResult:
    """Run a flow using tool-call routing (mirrors SimpleFlowAgent).

    Uses ToolCallAdapter instead of TextAdapter — the LLM picks edges
    via OpenAI function_calling, same as production SimpleFlowAgent.

    Args:
        flow: A ConversationFlow, JSON string, or path to a JSON file.
        model_id: OpenAI model ID (e.g. "gpt-4o-mini").
        user_messages: Scripted user inputs fed one per turn.
        system_prompt: Optional override for the flow's system_prompt.
    """
    from superdialog.machine.adapters.toolcall_adapter import ToolCallAdapter

    loaded = _load_flow(flow)
    prompt = system_prompt or loaded.system_prompt

    adapter = ToolCallAdapter(model_id=model_id, system_prompt=prompt)
    machine = await DialogStateMachine.from_flow(loaded, adapter)
    adapter._machine = machine

    responses: list[str] = []
    for msg in user_messages:
        if machine.is_complete:
            break
        result = await machine.process_turn(msg)
        if result.response:
            responses.append(result.response)

    return FlowResult(
        final_state=machine.current_state,
        is_complete=machine.is_complete,
        transitions=list(machine.context.data.transition_log),
        conversation_history=list(machine.context.data.history),
        responses=responses,
        userdata=dict(machine.context.data.variables),
    )


async def run_flow_from_node_toolcall(
    flow: ConversationFlow | str,
    model_id: str,
    start_node: str,
    user_messages: list[str],
    system_prompt: str = "",
) -> FlowResult:
    """Run a flow from a specific node using tool-call routing.

    Combines direct state injection with ToolCallAdapter.

    Args:
        flow: A ConversationFlow, JSON string, or path to a JSON file.
        model_id: OpenAI model ID (e.g. "gpt-4o-mini").
        start_node: Node ID to start execution from.
        user_messages: Scripted user inputs fed one per turn.
        system_prompt: Optional override for the flow's system_prompt.
    """
    from superdialog.machine.adapters.toolcall_adapter import ToolCallAdapter

    loaded = _load_flow(flow)
    node_ids = {n.id for n in loaded.nodes}
    if start_node not in node_ids:
        msg = f"Node '{start_node}' not found in flow"
        raise ValueError(msg)

    prompt = system_prompt or loaded.system_prompt

    adapter = ToolCallAdapter(model_id=model_id, system_prompt=prompt)
    machine = await DialogStateMachine.from_flow(loaded, adapter)
    adapter._machine = machine

    # Direct state injection
    if start_node != loaded.initial_node:
        machine.state = start_node
        machine.context.state.current_node_id = start_node
        machine.context.state.visit_count[start_node] = (
            machine.context.state.visit_count.get(start_node, 0) + 1
        )

    responses: list[str] = []
    for msg in user_messages:
        if machine.is_complete:
            break
        result = await machine.process_turn(msg)
        if result.response:
            responses.append(result.response)

    return FlowResult(
        final_state=machine.current_state,
        is_complete=machine.is_complete,
        transitions=list(machine.context.data.transition_log),
        conversation_history=list(machine.context.data.history),
        responses=responses,
        userdata=dict(machine.context.data.variables),
    )
