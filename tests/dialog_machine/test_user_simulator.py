"""Tests for UserSimulator -- persona-driven dialog flow simulation."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

for _mod in [
    "livekit.agents",
    "livekit.agents.llm",
    "livekit.agents.llm.tool_context",
    "livekit.agents.voice",
    "livekit.api",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402

from superdialog.flow.models import ConversationFlow, Edge, FlowNode  # noqa: E402
from superdialog.machine.eval.models import PersonaConfig, PersonaResult  # noqa: E402
from superdialog.machine.eval.user_simulator import (  # noqa: E402
    DEFAULT_PERSONAS,
    UserSimulator,
)


def _greeting_flow() -> ConversationFlow:
    """Simple flow: greeting -> collect_info -> goodbye."""
    return ConversationFlow(
        system_prompt="You are a friendly assistant.",
        initial_node="greeting",
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                instruction="Greet the user and ask for name.",
                edges=[
                    Edge(
                        id="name_given",
                        condition="User provides name",
                        target_node_id="collect_info",
                    ),
                ],
            ),
            FlowNode(
                id="collect_info",
                name="Collect Info",
                instruction="Thank them and say goodbye.",
                edges=[
                    Edge(
                        id="done",
                        condition="User confirms",
                        target_node_id="goodbye",
                    ),
                ],
            ),
            FlowNode(
                id="goodbye",
                name="Goodbye",
                static_text="Goodbye!",
                is_final=True,
            ),
        ],
    )


def _make_system_llm(edge_sequence: list[str]):
    """Mock system LLM that returns edges in sequence."""
    idx = {"i": 0}

    async def llm(messages: list[dict]) -> str:
        sys_content = messages[0].get("content", "")
        if "evaluating" in sys_content:
            edge_id = edge_sequence[idx["i"]] if idx["i"] < len(edge_sequence) else None
            idx["i"] += 1
            return json.dumps(
                {
                    "all_required_met": True,
                    "recommended_edge_id": edge_id,
                    "reason": "mock",
                }
            )
        return "Hello! What's your name?"

    return llm


def _make_persona_llm(responses: list[str]):
    """Mock persona LLM that returns canned responses."""
    idx = {"i": 0}

    async def llm(messages: list[dict]) -> str:
        resp = responses[idx["i"]] if idx["i"] < len(responses) else "Okay, bye"
        idx["i"] += 1
        return resp

    return llm


class TestUserSimulator:
    @pytest.mark.anyio
    async def test_cooperative_persona_reaches_final(self) -> None:
        flow = _greeting_flow()
        system_llm = _make_system_llm(["name_given", "done"])
        persona_llm = _make_persona_llm(["My name is Alice", "Yes, that's correct"])

        simulator = UserSimulator(
            flow=flow,
            system_llm_fn=system_llm,
            persona_llm_fn=persona_llm,
        )

        persona = PersonaConfig(
            name="cooperative",
            traits="Helpful and direct",
            goal="Complete the flow",
            max_turns=5,
        )

        result = await simulator.simulate(persona)
        assert isinstance(result, PersonaResult)
        assert result.final_node == "goodbye"
        assert result.reached_final is True
        assert result.turns_taken == 2
        assert len(result.transitions) == 2

    @pytest.mark.anyio
    async def test_max_turns_limits_simulation(self) -> None:
        """Simulation should stop at max_turns even if not complete."""
        flow = _greeting_flow()

        # LLM never triggers a transition
        async def no_transition_llm(messages: list[dict]) -> str:
            sys_content = messages[0].get("content", "")
            if "evaluating" in sys_content:
                return json.dumps(
                    {
                        "all_required_met": False,
                        "recommended_edge_id": None,
                        "reason": "not ready",
                    }
                )
            return "Can you tell me more?"

        persona_llm = _make_persona_llm(["hmm", "not sure", "maybe"])

        simulator = UserSimulator(
            flow=flow,
            system_llm_fn=no_transition_llm,
            persona_llm_fn=persona_llm,
        )

        persona = PersonaConfig(
            name="confused",
            traits="Unsure, gives vague answers",
            goal="Not sure what to do",
            max_turns=3,
        )

        result = await simulator.simulate(persona)
        assert result.turns_taken <= 3
        assert result.reached_final is False

    @pytest.mark.anyio
    async def test_persona_with_expected_final_node(self) -> None:
        """When expected_final_node is set, validate it matches."""
        flow = _greeting_flow()
        system_llm = _make_system_llm(["name_given", "done"])
        persona_llm = _make_persona_llm(["Alice", "Yes"])

        simulator = UserSimulator(
            flow=flow,
            system_llm_fn=system_llm,
            persona_llm_fn=persona_llm,
        )

        # Correct expected final
        persona = PersonaConfig(
            name="test",
            traits="Direct",
            goal="Complete",
            expected_final_node="goodbye",
            max_turns=5,
        )
        result = await simulator.simulate(persona)
        assert result.reached_final is True

    @pytest.mark.anyio
    async def test_conversation_history_recorded(self) -> None:
        flow = _greeting_flow()
        system_llm = _make_system_llm(["name_given"])
        persona_llm = _make_persona_llm(["My name is Bob"])

        simulator = UserSimulator(
            flow=flow,
            system_llm_fn=system_llm,
            persona_llm_fn=persona_llm,
        )

        persona = PersonaConfig(
            name="test",
            traits="",
            goal="",
            max_turns=2,
        )

        result = await simulator.simulate(persona)
        assert len(result.conversation) >= 2
        roles = [c["role"] for c in result.conversation]
        assert "assistant" in roles
        assert "user" in roles


class TestDefaultPersonas:
    def test_six_default_personas(self) -> None:
        assert len(DEFAULT_PERSONAS) == 6

    def test_all_personas_have_names(self) -> None:
        for p in DEFAULT_PERSONAS:
            assert p.name
            assert p.traits
            assert p.goal
