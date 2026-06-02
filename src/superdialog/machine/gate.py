"""TransitionGate — validates transitions before execution."""

from __future__ import annotations

import logging
import re
from typing import Any

from superdialog.flow.models import Edge, FlowNode
from superdialog.machine.models import FlowContext, TransitionResult

logger = logging.getLogger(__name__)

# Regex for detecting auto-proceed nodes
_AUTO_PROCEED_RE = re.compile(
    r"proceed immediately|no caller response needed|"
    r"no response needed|proceed directly",
    re.IGNORECASE,
)

# Fallback regex for flows that predate the node_type field.
# New flows should set node_type explicitly in the JSON instead.
# More specific pattern to avoid misclassifying collection nodes
_TOOL_ONLY_ROUTER_RE = re.compile(
    r"do not output any text|do not generate any speech|do not speak.*at all|"
    r"silent routing node|output only the tool call|"
    r"your only output must be a tool call|zero words.*before.*tool|just route.*no.*speech",
    re.IGNORECASE,
)


def classify_node_type(node: FlowNode) -> str:
    """Classify node as final/static/instruction/router.

    Priority:
    1. node.is_final
    2. node.node_type explicit field (set in flow JSON — preferred)
    3. Structural signals (static_text, instruction)
    4. Regex fallback for legacy flows without node_type set
    """
    if node.is_final:
        return "final"
    if node.node_type:
        return node.node_type
    if node.static_text and not node.instruction:
        return "static"
    if node.instruction:
        # Legacy fallback: detect silent-router pattern from instruction text
        if (
            not node.static_text
            and node.edges
            and _TOOL_ONLY_ROUTER_RE.search(node.instruction)
        ):
            return "router"
        return "instruction"
    return "router"


def is_auto_proceed(node: FlowNode) -> bool:
    """Check if node should auto-proceed without user input."""
    if node.auto_proceed:
        return True
    # Backward compatibility for flows that predate the auto_proceed field.
    return bool(node.instruction and _AUTO_PROCEED_RE.search(node.instruction))


class TransitionGate:
    """Validates whether a transition should be allowed.

    Gate checks (in order):
    1. Edge valid from current state
    2. Node content has been spoken
    2B. Self-loop limit
    3. Completion criteria (slot check)
    4A. User must have spoken
    4B. CriteriaJudge validation
    """

    async def check(
        self,
        edge_id: str,
        node: FlowNode,
        context: FlowContext,
        available_triggers: list[str],
        edge_obj: Edge | None,
        adapter: Any | None = None,
        collected_data: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Run all gates. Return allowed/denied with reason."""
        from_node = context.current_node_id

        # Gate 1: Edge valid?
        if edge_id not in available_triggers:
            reason = (
                f"Edge '{edge_id}' not available from node "
                f"'{from_node}'. Available: {available_triggers}"
            )
            logger.warning("[FLOW] gate DENIED (invalid edge): %s", reason)
            return TransitionResult(
                allowed=False,
                reason=reason,
                correction_hint=(
                    "The requested transition is not available. "
                    "Continue the conversation on the current topic."
                ),
            )

        # Gate 2: Node content spoken?
        if not context.node_spoken:
            reason = (
                f"Node '{from_node}' content has not been spoken yet. "
                "Executor must deliver the node's message before "
                "transitioning."
            )
            logger.warning("[FLOW] gate DENIED (not spoken): %s", reason)
            return TransitionResult(
                allowed=False,
                reason=reason,
                correction_hint=(
                    "You must first deliver the message for this step "
                    "before moving to the next step. Speak the content, "
                    "then try again."
                ),
            )

        # Gate 2B: Self-loop protection
        is_self_loop = edge_obj is not None and edge_obj.target_node_id == from_node
        if is_self_loop and context.consecutive_self_loops >= context.MAX_SELF_LOOPS:
            reason = (
                f"Self-loop limit reached for node '{from_node}' "
                f"({context.consecutive_self_loops} consecutive). "
                "Choose a different transition."
            )
            logger.warning("[FLOW] gate DENIED (self-loop limit): %s", reason)
            return TransitionResult(
                allowed=False,
                reason=reason,
                correction_hint=(
                    "You have already looped back to this step "
                    f"{context.consecutive_self_loops} times. "
                    "You MUST choose a different transition tool "
                    "that moves the conversation forward."
                ),
            )

        # Gate 3: Completion criteria (lightweight slot check)
        #
        # Data is checked in layers (later layers override earlier):
        #   1. context.userdata   — data collected in *previous* nodes
        #   2. node_slots         — data accumulated in this node
        #   3. collected_data     — data from the current transition
        #
        # This ensures that information collected earlier (e.g. city
        # collected during greeting) satisfies criteria in later nodes
        # (e.g. collect_booking_details) without forcing the LLM to
        # re-pass it in every tool call.
        gate3_all_slots_filled = False
        if node.completion_criteria:
            current_slots = context.node_slots.get(from_node, {})
            check_slots = dict(context.userdata)
            check_slots.update(current_slots)
            if collected_data:
                check_slots.update(collected_data)

            missing: list[str] = []
            for criterion in node.completion_criteria:
                if not criterion.required:
                    continue
                val = check_slots.get(criterion.key)
                if val is None or (isinstance(val, str) and not val.strip()):
                    missing.append(f"{criterion.key}: {criterion.description}")

            if missing and not node.allow_skip:
                reason = (
                    f"Required criteria not met for node "
                    f"'{from_node}': " + "; ".join(missing)
                )
                logger.warning("[FLOW] gate DENIED (criteria): %s", reason)
                return TransitionResult(
                    allowed=False,
                    reason=reason,
                    missing_criteria=missing,
                    correction_hint=(
                        "The following information is still needed "
                        "before moving on: "
                        + "; ".join(missing)
                        + ". Please collect this from the caller."
                    ),
                    recovery_speech=(
                        "I still need a few details from you before we can proceed."
                    ),
                )

            if not missing:
                gate3_all_slots_filled = True

        # Gate 4A: User must have spoken
        node_type = classify_node_type(node)
        _auto = is_auto_proceed(node)
        # Router/static nodes route on already-available context or scripted
        # speech — no new user input is required to leave them. Instruction
        # nodes still need a user turn before transitioning.
        needs_user_input = not _auto and node_type not in (
            "final",
            "router",
            "static",
        )
        if needs_user_input and not context.user_spoke_in_node:
            reason = (
                f"No user input received in node '{from_node}'. "
                "The caller must respond before transitioning."
            )
            logger.warning("[FLOW] gate DENIED (no user input): %s", reason)
            # Only save as pending for auto-retry if collected_data
            # already contains the required fields.  If the edge has
            # required input_schema fields and collected_data is empty,
            # replaying after the user speaks would just fire the
            # action with missing data (e.g. course_id, date).
            _has_required = False
            if edge_obj and isinstance(edge_obj.input_schema, dict):
                required = edge_obj.input_schema.get("required", [])
                if required:
                    _has_required = True
                    supplied = set(collected_data.keys()) if collected_data else set()
                    missing_required = set(required) - supplied
                    if missing_required:
                        logger.warning(
                            "[FLOW] gate 4A: NOT saving pending edge=%s "
                            "— collected_data missing required fields %s",
                            edge_id,
                            sorted(missing_required),
                        )
                        return TransitionResult(
                            allowed=False,
                            reason=reason,
                            correction_hint=(
                                "You must wait for the caller to respond "
                                "and collect the required data before "
                                "calling this tool. Missing fields: "
                                f"{sorted(missing_required)}. Do NOT call "
                                "this tool until you have all required data."
                            ),
                            recovery_speech="I'm listening. Please go ahead.",
                        )

            # Remember this edge so the machine can auto-retry it once
            # the user speaks — no need for the LLM to call the tool again.
            # Also save collected_data so slot_id / player_id etc. survive.
            context.pending_edge_id = edge_id
            context.pending_collected_data = collected_data
            logger.info(
                "[FLOW] gate 4A: saved pending_edge_id=%s collected=%s for node=%s",
                edge_id,
                list(collected_data.keys()) if collected_data else [],
                from_node,
            )
            return TransitionResult(
                allowed=False,
                reason=reason,
                correction_hint=(
                    "You must wait for the caller to respond before "
                    "calling a transition tool. Do NOT transition "
                    "until the caller has spoken."
                ),
                recovery_speech="I'm listening. Please go ahead.",
            )

        # Gate 4B: CriteriaJudge validation
        _adapter_supports_criteria = getattr(adapter, "supports_criteria", True)
        if (
            node.completion_criteria
            and adapter
            and _adapter_supports_criteria
            and not gate3_all_slots_filled
        ):
            try:
                eval_userdata = {
                    **context.userdata,
                    "_flow_meta": {
                        "visit_count": context.visit_count.get(from_node, 1),
                        "turns_in_node": context.turns_in_node,
                        "user_turns_in_node": (context.user_turns_in_node),
                        "agent_language": context.agent_language,
                        "agent_gender": context.agent_gender,
                        "node_slots": context.node_slots.get(from_node, {}),
                        "previously_completed": (from_node in context.completed_nodes),
                    },
                }
                judge_result = await adapter.evaluate_criteria(
                    node,
                    context.conversation_history,
                    eval_userdata,
                )

                if (
                    judge_result.recommended_edge_id
                    and judge_result.recommended_edge_id != edge_id
                ):
                    logger.warning(
                        "[FLOW] gate 4B: judge recommends edge=%s "
                        "but LLM requested edge=%s (allowing LLM)",
                        judge_result.recommended_edge_id,
                        edge_id,
                    )

                has_criteria = bool(node.completion_criteria)
                if (
                    has_criteria
                    and not judge_result.all_required_met
                    and not node.allow_skip
                ):
                    reason = (
                        f"CriteriaJudge says requirements not met "
                        f"for node '{from_node}': "
                        f"{judge_result.reason}"
                    )
                    logger.warning("[FLOW] gate DENIED (judge): %s", reason)
                    hint = judge_result.response or (
                        "Some required information is still "
                        "missing. Please continue the conversation "
                        "to collect it."
                    )
                    return TransitionResult(
                        allowed=False,
                        reason=reason,
                        correction_hint=hint,
                    )

            except Exception as exc:
                logger.warning(
                    "[FLOW] gate 4B judge failed (non-fatal): %s",
                    exc,
                )

        # All gates passed
        return TransitionResult(allowed=True)
