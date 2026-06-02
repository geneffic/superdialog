"""TextAdapter -- provider-agnostic adapter for text/chat flows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from superdialog.machine.criteria import CriteriaJudge, LLMCallable
from superdialog.machine.models import CriteriaResult

if TYPE_CHECKING:
    from superdialog.flow.models import CustomAction, FlowNode

logger = logging.getLogger(__name__)


class TextAdapter:
    """Runtime adapter for text-based flow execution.

    Uses any LLM via a simple async callable:
        async def llm(messages: list[dict]) -> str

    Responses accumulate in ``self.responses`` for inspection.
    """

    def __init__(
        self,
        llm_fn: LLMCallable,
        criteria_judge: CriteriaJudge | None = None,
        system_prompt: str = "",
    ) -> None:
        self._llm_fn = llm_fn
        self._judge = criteria_judge or CriteriaJudge(llm_fn=llm_fn)
        self._system_prompt = system_prompt
        self.responses: list[str] = []
        self.session_ended: bool = False

    async def speak(self, text: str, node: FlowNode) -> None:
        """Record static text as a response."""
        self.responses.append(text)

    async def generate_reply(
        self,
        instruction: str,
        node: FlowNode,
        history: list[dict[str, Any]] | None = None,
        userdata: dict[str, Any] | None = None,
    ) -> str:
        """Generate LLM reply from instruction."""
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (f"{self._system_prompt}\n\n{instruction}"),
            },
        ]
        if history:
            messages.extend(history)
        reply = await self._llm_fn(messages)
        self.responses.append(reply)
        return reply

    async def evaluate_criteria(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
    ) -> CriteriaResult:
        """Evaluate node completion via CriteriaJudge."""
        # Extract flow metadata injected by DialogStateMachine
        meta = userdata.get("_flow_meta", {})
        # Pass clean userdata to judge (without internal metadata)
        clean_userdata = {k: v for k, v in userdata.items() if k != "_flow_meta"}
        return await self._judge.evaluate(
            node=node,
            history=history,
            userdata=clean_userdata,
            system_prompt=self._system_prompt,
            visit_count=meta.get("visit_count", 1),
            turns_in_node=meta.get("turns_in_node", 0),
            agent_language=meta.get("agent_language", ""),
            agent_gender=meta.get("agent_gender", ""),
            node_slots=meta.get("node_slots"),
            previously_completed=meta.get("previously_completed", False),
        )

    async def generate_recovery(self, node: FlowNode, error: str) -> str:
        """Generate a recovery response for the user."""
        instruction = node.instruction or node.static_text or ""
        fallback = "I didn't quite catch that. Could you say that again?"
        if not instruction:
            return fallback
        try:
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        f"{self._system_prompt}\n\n"
                        f"The previous response failed to process. "
                        f"Generate a brief, natural recovery message "
                        f"that re-engages the user with the current "
                        f"task: {instruction}"
                    ),
                },
            ]
            return await self._llm_fn(messages)
        except Exception:
            return fallback

    async def execute_action(
        self,
        action: CustomAction,
        userdata: dict[str, Any],
    ) -> dict[str, Any] | None:
        """No-op for text adapter (actions not supported)."""
        logger.info("TextAdapter: skipping action %s", action.id)
        return None

    async def end_session(self) -> None:
        """Mark session as ended."""
        self.session_ended = True
