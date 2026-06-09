# src/superdialog/machine/eval/user_simulator.py
from __future__ import annotations

import logging
from typing import Any, Callable

from superdialog.flow.models import ConversationFlow
from superdialog.machine.criteria import CriteriaJudge
from superdialog.machine.eval.models import PersonaConfig, PersonaResult
from superdialog.machine.models import TransitionRecord

logger = logging.getLogger(__name__)

LLMFn = Callable[[list[dict[str, Any]]], Any]

DEFAULT_PERSONAS: list[PersonaConfig] = [
    PersonaConfig(
        name="cooperative",
        traits="Direct, helpful, provides complete information",
        goal="Complete the conversation successfully",
        max_turns=10,
    ),
    PersonaConfig(
        name="hesitant",
        traits="Unsure, gives vague or incomplete answers, needs prompting",
        goal="Eventually complete the flow but with multiple retries",
        max_turns=15,
    ),
    PersonaConfig(
        name="impatient",
        traits="Wants to rush through, skips details, says 'yes' to everything",
        goal="Finish as fast as possible",
        max_turns=5,
    ),
    PersonaConfig(
        name="off-topic",
        traits="Frequently goes off-script, asks unrelated questions",
        goal="Talk about unrelated things then eventually complete the flow",
        max_turns=12,
    ),
    PersonaConfig(
        name="hindi-speaker",
        traits="Prefers to speak in Hindi or Hinglish",
        goal="Complete the flow while speaking Hindi",
        max_turns=10,
    ),
    PersonaConfig(
        name="aggressive",
        traits="Pushes back, challenges the agent, refuses to provide information initially",
        goal="Eventually comply after being convinced",
        max_turns=15,
    ),
]


class UserSimulator:
    def __init__(
        self,
        flow: ConversationFlow,
        system_llm_fn: LLMFn,
        persona_llm_fn: LLMFn,
    ) -> None:
        self._flow = flow
        self._system_llm_fn = system_llm_fn
        self._persona_llm_fn = persona_llm_fn
        self._node_map = {n.id: n for n in flow.nodes}

    async def simulate(self, persona: PersonaConfig) -> PersonaResult:
        judge = CriteriaJudge(llm_fn=self._system_llm_fn)
        current_node_id = self._flow.initial_node
        current_node = self._node_map.get(current_node_id)
        conversation: list[dict[str, str]] = []
        transitions: list[dict[str, Any]] = []
        turns = 0

        if current_node is not None:
            opening = await self._system_llm_fn([
                {"role": "system", "content": current_node.instruction or current_node.static_text or ""},
                {"role": "user", "content": "[START CONVERSATION]"},
            ])
            conversation.append({"role": "assistant", "content": opening})

        history: list[dict[str, Any]] = list(conversation)

        while turns < persona.max_turns and current_node is not None:
            if current_node.is_final:
                break

            persona_messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a caller with these traits: {persona.traits}. "
                        f"Your goal: {persona.goal}. "
                        f"Reply naturally to the agent's last message. Keep it short (1-2 sentences)."
                    ),
                },
                *history,
            ]
            user_reply = await self._persona_llm_fn(persona_messages)
            conversation.append({"role": "user", "content": user_reply})
            history.append({"role": "user", "content": user_reply})
            turns += 1

            result = await judge.evaluate(
                node=current_node,
                history=history,
                userdata={},
                system_prompt=self._flow.system_prompt or "",
            )

            if result.recommended_edge_id:
                edge = next(
                    (e for e in current_node.edges if e.id == result.recommended_edge_id),
                    None,
                )
                if edge and edge.target_node_id:
                    record = TransitionRecord(
                        from_node=current_node_id,
                        to_node=edge.target_node_id,
                        edge_id=edge.id,
                    )
                    transitions.append(record.model_dump())
                    current_node_id = edge.target_node_id
                    current_node = self._node_map.get(current_node_id)

                    if current_node is None or current_node.is_final:
                        break

                    agent_msg = await self._system_llm_fn([
                        {"role": "system", "content": current_node.instruction or current_node.static_text or ""},
                        *history,
                    ])
                    conversation.append({"role": "assistant", "content": agent_msg})
                    history.append({"role": "assistant", "content": agent_msg})

        reached_final = (
            current_node is not None and current_node.is_final
        ) or (
            persona.expected_final_node is not None
            and current_node_id == persona.expected_final_node
        )

        return PersonaResult(
            persona_name=persona.name,
            model_id="",
            final_node=current_node_id,
            expected_final_node=persona.expected_final_node,
            reached_final=reached_final,
            turns_taken=turns,
            transitions=transitions,
            conversation=conversation,
        )