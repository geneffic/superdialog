"""Tests for gated node traversal — NodeScope, request_transition, gates.

Verifies:
  1. NodeScope is correctly built with all data (history, tools, criteria)
  2. Gate checks: spoken flag, criteria validation, edge validity
  3. Full traversal flow with gate enforcement
  4. Denied transitions return correction hints
  5. node_spoken tracking across transitions
"""

from __future__ import annotations

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

from superdialog.flow.models import (  # noqa: E402
    CompletionCriterion,
    ConversationFlow,
    Edge,
    FlowNode,
)
from superdialog.machine.machine import DialogStateMachine  # noqa: E402
from superdialog.machine.models import FlowContext  # noqa: E402
from superdialog.machine.testing.mock_adapter import MockAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Test flows
# ---------------------------------------------------------------------------


def _simple_flow() -> ConversationFlow:
    """3-node flow: start -> middle -> end."""
    return ConversationFlow(
        system_prompt="You are a test agent.",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                static_text="Hello!",
                edges=[
                    Edge(
                        id="e_start_mid",
                        condition="user ready",
                        target_node_id="middle",
                    ),
                ],
            ),
            FlowNode(
                id="middle",
                name="Middle",
                instruction="Ask user something",
                edges=[
                    Edge(
                        id="e_mid_end",
                        condition="user done",
                        target_node_id="end",
                    ),
                ],
            ),
            FlowNode(
                id="end",
                name="End",
                static_text="Goodbye!",
                is_final=True,
            ),
        ],
    )


def _flow_with_criteria() -> ConversationFlow:
    """3-node flow where middle node has required completion criteria."""
    return ConversationFlow(
        system_prompt="You are a data collector.",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                static_text="Welcome!",
                edges=[
                    Edge(
                        id="e_start_collect",
                        condition="begin collection",
                        target_node_id="collect",
                    ),
                ],
            ),
            FlowNode(
                id="collect",
                name="Collect Info",
                instruction="Ask for the caller's name and budget.",
                completion_criteria=[
                    CompletionCriterion(
                        key="name",
                        description="Caller's full name",
                        required=True,
                    ),
                    CompletionCriterion(
                        key="budget",
                        description="Caller's budget amount",
                        required=True,
                    ),
                    CompletionCriterion(
                        key="notes",
                        description="Optional notes",
                        required=False,
                    ),
                ],
                allow_skip=False,
                edges=[
                    Edge(
                        id="e_collect_done",
                        condition="info collected",
                        target_node_id="done",
                    ),
                ],
            ),
            FlowNode(
                id="done",
                name="Done",
                static_text="Thank you!",
                is_final=True,
            ),
        ],
    )


def _flow_with_branching() -> ConversationFlow:
    """Flow with branching: start -> {yes_path, no_path} -> end."""
    return ConversationFlow(
        system_prompt="You are a router.",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="[EN] Do you want to continue?",
                edges=[
                    Edge(
                        id="e_yes",
                        condition="user says yes",
                        target_node_id="yes_path",
                    ),
                    Edge(
                        id="e_no",
                        condition="user says no",
                        target_node_id="no_path",
                    ),
                ],
            ),
            FlowNode(
                id="yes_path",
                name="Yes Path",
                static_text="Great, continuing!",
                edges=[
                    Edge(
                        id="e_yes_end",
                        condition="done",
                        target_node_id="end",
                    ),
                ],
            ),
            FlowNode(
                id="no_path",
                name="No Path",
                static_text="Okay, goodbye!",
                edges=[
                    Edge(
                        id="e_no_end",
                        condition="done",
                        target_node_id="end",
                    ),
                ],
            ),
            FlowNode(
                id="end",
                name="End",
                static_text="Call ended.",
                is_final=True,
            ),
        ],
    )


async def _make_machine(
    flow: ConversationFlow,
    userdata: dict | None = None,
) -> DialogStateMachine:
    """Create a machine with a mock adapter."""
    adapter = MockAdapter(edge_sequence=[])
    machine = await DialogStateMachine.from_flow(flow, adapter)
    if userdata:
        machine.context.userdata.update(userdata)
    return machine


# ===========================================================================
# Test: NodeScope building
# ===========================================================================


class TestBuildNodeScope:
    """Verify build_node_scope() assembles all required data."""

    @pytest.mark.anyio
    async def test_scope_has_node_identity(self) -> None:
        machine = await _make_machine(_simple_flow())
        scope = machine.build_node_scope()

        assert scope.node_id == "start"
        assert scope.node_type == "static"
        assert scope.is_initial is True
        assert scope.is_final is False
        assert scope.auto_proceed is False

    @pytest.mark.anyio
    async def test_scope_has_system_prompt(self) -> None:
        machine = await _make_machine(_simple_flow())
        scope = machine.build_node_scope()

        assert scope.system_prompt == "You are a test agent."

    @pytest.mark.anyio
    async def test_scope_has_enriched_instructions(self) -> None:
        machine = await _make_machine(_simple_flow())
        scope = machine.build_node_scope()

        assert scope.node_instruction  # non-empty
        assert "Hello!" in scope.node_instruction  # static_text included

    @pytest.mark.anyio
    async def test_scope_has_conversation_history(self) -> None:
        machine = await _make_machine(_simple_flow())
        # Add some history
        machine.context.add_user_message("hi there")
        machine.context.add_assistant_message("hello!")

        scope = machine.build_node_scope()

        assert len(scope.conversation_history) == 2
        assert scope.conversation_history[0]["role"] == "user"
        assert scope.conversation_history[1]["role"] == "assistant"

    @pytest.mark.anyio
    async def test_scope_has_edge_tools(self) -> None:
        machine = await _make_machine(_simple_flow())
        scope = machine.build_node_scope()

        assert len(scope.edge_tools) >= 1
        tool_ids = [t.id for t in scope.edge_tools]
        assert "e_start_mid" in tool_ids

    @pytest.mark.anyio
    async def test_scope_has_userdata(self) -> None:
        machine = await _make_machine(
            _simple_flow(),
            userdata={"name": "Rahul", "budget": "10000"},
        )
        scope = machine.build_node_scope()

        assert scope.userdata["name"] == "Rahul"
        assert scope.userdata["budget"] == "10000"

    @pytest.mark.anyio
    async def test_scope_has_completion_criteria(self) -> None:
        machine = await _make_machine(_flow_with_criteria())
        # Move to the collect node
        machine.mark_node_spoken()
        await machine.apply_transition("e_start_collect")

        scope = machine.build_node_scope()

        assert scope.node_id == "collect"
        assert len(scope.completion_criteria) == 3
        required = [c for c in scope.completion_criteria if c["required"]]
        assert len(required) == 2
        keys = [c["key"] for c in required]
        assert "name" in keys
        assert "budget" in keys

    @pytest.mark.anyio
    async def test_scope_tracks_visit_count(self) -> None:
        machine = await _make_machine(_simple_flow())
        scope = machine.build_node_scope()
        assert scope.visit_count == 1

    @pytest.mark.anyio
    async def test_scope_after_transition_has_history(self) -> None:
        """After transitioning, the new scope includes prior history."""
        machine = await _make_machine(_simple_flow())
        machine.context.add_user_message("hello")
        machine.context.add_assistant_message("hi!")
        machine.mark_node_spoken()

        await machine.apply_transition("e_start_mid")

        scope = machine.build_node_scope()
        assert scope.node_id == "middle"
        assert scope.is_initial is False
        # History from previous node is preserved
        assert len(scope.conversation_history) >= 2
        assert "start" in scope.completed_nodes

    @pytest.mark.anyio
    async def test_scope_for_final_node(self) -> None:
        machine = await _make_machine(_simple_flow())
        machine.mark_node_spoken()
        await machine.apply_transition("e_start_mid")
        machine.mark_node_spoken()
        await machine.apply_transition("e_mid_end")

        scope = machine.build_node_scope()
        assert scope.node_id == "end"
        assert scope.is_final is True
        assert scope.node_type == "final"


# ===========================================================================
# Test: node_spoken tracking
# ===========================================================================


class TestNodeSpokenTracking:
    """Verify the machine tracks which nodes have been spoken."""

    @pytest.mark.anyio
    async def test_node_not_spoken_initially(self) -> None:
        machine = await _make_machine(_simple_flow())
        assert machine.context.node_spoken is False

    @pytest.mark.anyio
    async def test_mark_node_spoken(self) -> None:
        machine = await _make_machine(_simple_flow())
        machine.mark_node_spoken()
        assert machine.context.node_spoken is True

    @pytest.mark.anyio
    async def test_spoken_resets_on_new_node(self) -> None:
        machine = await _make_machine(_simple_flow())
        machine.mark_node_spoken()
        assert machine.context.node_spoken is True

        await machine.apply_transition("e_start_mid")
        # New node should not be marked as spoken
        assert machine.context.node_spoken is False

    @pytest.mark.anyio
    async def test_spoken_flag_per_node(self) -> None:
        machine = await _make_machine(_simple_flow())
        machine.mark_node_spoken("start")

        # start is spoken, middle is not
        assert machine.context.node_spoken_flags.get("start") is True
        assert machine.context.node_spoken_flags.get("middle") is None

    @pytest.mark.anyio
    async def test_spoken_excluded_from_serialization(self) -> None:
        ctx = FlowContext(current_node_id="test")
        ctx.node_spoken = True
        dumped = ctx.model_dump()
        assert "node_spoken_flags" not in dumped


# ===========================================================================
# Test: request_transition gate checks
# ===========================================================================


class TestRequestTransitionGates:
    """Verify the gate checks in request_transition()."""

    @pytest.mark.anyio
    async def test_gate_denies_invalid_edge(self) -> None:
        machine = await _make_machine(_simple_flow())
        machine.mark_node_spoken()

        result = await machine.request_transition("nonexistent_edge")

        assert result.allowed is False
        assert "not available" in result.reason
        assert result.correction_hint  # has guidance

    @pytest.mark.anyio
    async def test_gate_denies_when_not_spoken(self) -> None:
        machine = await _make_machine(_simple_flow())
        # Do NOT mark as spoken

        result = await machine.request_transition("e_start_mid")

        assert result.allowed is False
        assert "not been spoken" in result.reason
        assert "deliver the message" in result.correction_hint

    @pytest.mark.anyio
    async def test_gate_allows_after_spoken(self) -> None:
        machine = await _make_machine(_simple_flow())
        machine.mark_node_spoken()

        result = await machine.request_transition("e_start_mid")

        assert result.allowed is True
        assert result.turn_result is not None
        assert result.turn_result.from_node == "start"
        assert result.turn_result.to_node == "middle"
        assert result.new_scope is not None
        assert result.new_scope.node_id == "middle"

    @pytest.mark.anyio
    async def test_gate_denies_missing_criteria(self) -> None:
        machine = await _make_machine(_flow_with_criteria())
        machine.mark_node_spoken()
        await machine.apply_transition("e_start_collect")
        machine.mark_node_spoken()
        machine.context.add_user_message("I want to sign up")

        # Try to transition without providing required data
        result = await machine.request_transition("e_collect_done")

        assert result.allowed is False
        assert len(result.missing_criteria) > 0
        assert any("name" in mc for mc in result.missing_criteria)
        assert any("budget" in mc for mc in result.missing_criteria)
        assert "needed" in result.correction_hint

    @pytest.mark.anyio
    async def test_gate_allows_with_criteria_met(self) -> None:
        machine = await _make_machine(_flow_with_criteria())
        machine.mark_node_spoken()
        await machine.apply_transition("e_start_collect")
        machine.mark_node_spoken()
        machine.context.add_user_message("My name is Rahul, budget 10000")

        # Provide required data
        machine.context.node_slots.setdefault("collect", {}).update(
            {"name": "Rahul", "budget": "10000"}
        )
        machine.context.userdata.update({"name": "Rahul", "budget": "10000"})

        result = await machine.request_transition("e_collect_done")

        assert result.allowed is True
        assert result.turn_result is not None
        assert result.turn_result.to_node == "done"

    @pytest.mark.anyio
    async def test_gate_allows_with_collected_data_in_request(self) -> None:
        """Collected data passed to request_transition counts for criteria."""
        machine = await _make_machine(_flow_with_criteria())
        machine.mark_node_spoken()
        await machine.apply_transition("e_start_collect")
        machine.mark_node_spoken()
        machine.context.add_user_message("Rahul, budget 10000")

        result = await machine.request_transition(
            "e_collect_done",
            collected_data={"name": "Rahul", "budget": "10000"},
        )

        assert result.allowed is True

    @pytest.mark.anyio
    async def test_gate_allows_partial_criteria_when_skip_allowed(
        self,
    ) -> None:
        """If allow_skip=True, missing criteria don't block."""
        flow = _flow_with_criteria()
        # Change allow_skip to True
        for node in flow.nodes:
            if node.id == "collect":
                node.allow_skip = True

        machine = await _make_machine(flow)
        machine.mark_node_spoken()
        await machine.apply_transition("e_start_collect")
        machine.mark_node_spoken()
        machine.context.add_user_message("skip please")

        result = await machine.request_transition("e_collect_done")

        # Should be allowed even without data because allow_skip=True
        assert result.allowed is True

    @pytest.mark.anyio
    async def test_gate_optional_criteria_dont_block(self) -> None:
        """Optional (required=False) criteria don't block transition."""
        machine = await _make_machine(_flow_with_criteria())
        machine.mark_node_spoken()
        await machine.apply_transition("e_start_collect")
        machine.mark_node_spoken()
        machine.context.add_user_message("Rahul, budget 10000")

        # Provide only required fields, not 'notes' (optional)
        machine.context.node_slots.setdefault("collect", {}).update(
            {"name": "Rahul", "budget": "10000"}
        )
        machine.context.userdata.update({"name": "Rahul", "budget": "10000"})

        result = await machine.request_transition("e_collect_done")

        assert result.allowed is True  # 'notes' not required


# ===========================================================================
# Test: new scope after transition
# ===========================================================================


class TestNewScopeAfterTransition:
    """Verify the new_scope returned after a successful transition."""

    @pytest.mark.anyio
    async def test_new_scope_has_correct_node(self) -> None:
        machine = await _make_machine(_simple_flow())
        machine.mark_node_spoken()

        result = await machine.request_transition("e_start_mid")

        assert result.new_scope is not None
        assert result.new_scope.node_id == "middle"
        assert result.new_scope.node_type == "instruction"

    @pytest.mark.anyio
    async def test_new_scope_carries_history(self) -> None:
        machine = await _make_machine(_simple_flow())
        machine.context.add_user_message("I'm ready")
        machine.context.add_assistant_message("Great!")
        machine.mark_node_spoken()

        result = await machine.request_transition("e_start_mid")

        assert result.new_scope is not None
        # History should include the messages from start node
        assert len(result.new_scope.conversation_history) >= 2

    @pytest.mark.anyio
    async def test_new_scope_shows_completed_nodes(self) -> None:
        machine = await _make_machine(_simple_flow())
        machine.mark_node_spoken()

        result = await machine.request_transition("e_start_mid")

        assert result.new_scope is not None
        assert "start" in result.new_scope.completed_nodes

    @pytest.mark.anyio
    async def test_new_scope_has_tools_for_new_node(self) -> None:
        machine = await _make_machine(_simple_flow())
        machine.mark_node_spoken()

        result = await machine.request_transition("e_start_mid")

        assert result.new_scope is not None
        tool_ids = [t.id for t in result.new_scope.edge_tools]
        assert "e_mid_end" in tool_ids
        assert "e_start_mid" not in tool_ids  # old node's tool gone


# ===========================================================================
# Test: full traversal with gates
# ===========================================================================


class TestFullGatedTraversal:
    """End-to-end tests: traverse an entire flow using gated transitions."""

    @pytest.mark.anyio
    async def test_simple_flow_traversal(self) -> None:
        """Walk through start -> middle -> end with proper gating."""
        machine = await _make_machine(_simple_flow())

        # Node: start (static — no user input required by Gate 4A)
        scope = machine.build_node_scope()
        assert scope.node_id == "start"
        assert scope.is_initial is True

        machine.mark_node_spoken()
        r1 = await machine.request_transition("e_start_mid")
        assert r1.allowed is True

        # Node: middle (instruction — Gate 4A requires user input)
        assert r1.new_scope is not None
        assert r1.new_scope.node_id == "middle"

        machine.mark_node_spoken()
        machine.context.add_user_message("I'm done")
        r2 = await machine.request_transition("e_mid_end")
        assert r2.allowed is True

        # Node: end
        assert r2.new_scope is not None
        assert r2.new_scope.node_id == "end"
        assert r2.new_scope.is_final is True

        # Verify traversal log
        assert len(machine.context.transition_log) == 2
        assert machine.context.transition_log[0].from_node == "start"
        assert machine.context.transition_log[0].to_node == "middle"
        assert machine.context.transition_log[1].from_node == "middle"
        assert machine.context.transition_log[1].to_node == "end"

    @pytest.mark.anyio
    async def test_branching_flow_traversal(self) -> None:
        """Walk through branching flow: start -> yes_path -> end."""
        machine = await _make_machine(_flow_with_branching())

        scope = machine.build_node_scope()
        assert scope.node_id == "start"
        tool_ids = [t.id for t in scope.edge_tools]
        assert "e_yes" in tool_ids
        assert "e_no" in tool_ids

        # start is instruction — user must speak
        machine.mark_node_spoken()
        machine.context.add_user_message("yes")
        r1 = await machine.request_transition("e_yes")
        assert r1.allowed is True
        assert r1.new_scope is not None
        assert r1.new_scope.node_id == "yes_path"

        # yes_path is static — no user input needed
        machine.mark_node_spoken()
        r2 = await machine.request_transition("e_yes_end")
        assert r2.allowed is True
        assert r2.new_scope is not None
        assert r2.new_scope.node_id == "end"

    @pytest.mark.anyio
    async def test_criteria_flow_full_traversal(self) -> None:
        """Full flow with criteria: start -> collect (fill data) -> done."""
        machine = await _make_machine(_flow_with_criteria())

        # Start -> collect (static node, no user input needed)
        machine.mark_node_spoken()
        r1 = await machine.request_transition("e_start_collect")
        assert r1.allowed is True

        # Try to skip collect without data — user spoke but no slots
        machine.mark_node_spoken()
        machine.context.add_user_message("I want to sign up")
        r2 = await machine.request_transition("e_collect_done")
        assert r2.allowed is False
        assert len(r2.missing_criteria) == 2

        # Fill partial data
        machine.context.node_slots.setdefault("collect", {})["name"] = "Rahul"
        machine.context.userdata["name"] = "Rahul"

        r3 = await machine.request_transition("e_collect_done")
        assert r3.allowed is False
        assert len(r3.missing_criteria) == 1
        assert any("budget" in mc for mc in r3.missing_criteria)

        # Fill remaining data
        machine.context.node_slots["collect"]["budget"] = "10000"
        machine.context.userdata["budget"] = "10000"

        r4 = await machine.request_transition("e_collect_done")
        assert r4.allowed is True
        assert r4.new_scope is not None
        assert r4.new_scope.node_id == "done"
        assert r4.new_scope.is_final is True

    @pytest.mark.anyio
    async def test_skipping_not_spoken_blocks_traversal(self) -> None:
        """Attempting to skip a node without speaking blocks transition."""
        machine = await _make_machine(_simple_flow())

        # Try to go start -> mid without speaking
        r = await machine.request_transition("e_start_mid")
        assert r.allowed is False
        assert "not been spoken" in r.reason

        # Now speak and retry
        machine.mark_node_spoken()
        r = await machine.request_transition("e_start_mid")
        assert r.allowed is True

    @pytest.mark.anyio
    async def test_history_accumulates_across_nodes(self) -> None:
        """Conversation history grows as we traverse nodes."""
        machine = await _make_machine(_simple_flow())

        machine.context.add_assistant_message("Hello!")
        machine.context.add_user_message("Hi")
        machine.mark_node_spoken()

        r1 = await machine.request_transition("e_start_mid")
        assert r1.allowed is True
        assert r1.new_scope is not None

        # Add more history at middle node
        machine.context.add_assistant_message("What can I help with?")
        machine.context.add_user_message("I'm done")
        machine.mark_node_spoken()

        r2 = await machine.request_transition("e_mid_end")
        assert r2.allowed is True
        assert r2.new_scope is not None

        # Final scope should have ALL history
        assert len(r2.new_scope.conversation_history) >= 4


# ===========================================================================
# Test: Gate 4A — user must have spoken (intent check)
# ===========================================================================


class TestGate4AUserSpoke:
    """Gate 4A: instruction/router nodes require user input before transition."""

    @pytest.mark.anyio
    async def test_instruction_node_denies_without_user_input(self) -> None:
        """Instruction node blocks if user hasn't spoken."""
        machine = await _make_machine(_simple_flow())
        machine.mark_node_spoken()
        await machine.apply_transition("e_start_mid")

        # middle is instruction — user must speak
        machine.mark_node_spoken()
        # Do NOT add user message

        r = await machine.request_transition("e_mid_end")
        assert r.allowed is False
        assert "No user input" in r.reason
        assert "wait for the caller" in r.correction_hint

    @pytest.mark.anyio
    async def test_instruction_node_allows_after_user_speaks(self) -> None:
        machine = await _make_machine(_simple_flow())
        machine.mark_node_spoken()
        await machine.apply_transition("e_start_mid")

        machine.mark_node_spoken()
        machine.context.add_user_message("Yes, I'm ready")

        r = await machine.request_transition("e_mid_end")
        assert r.allowed is True

    @pytest.mark.anyio
    async def test_static_node_allows_without_user_input(self) -> None:
        """Static nodes (greeting) don't require user input."""
        machine = await _make_machine(_simple_flow())

        # start is static — should allow without user message
        machine.mark_node_spoken()
        r = await machine.request_transition("e_start_mid")
        assert r.allowed is True

    @pytest.mark.anyio
    async def test_auto_proceed_allows_without_user_input(self) -> None:
        """Auto-proceed nodes skip user-spoke check."""
        flow = ConversationFlow(
            system_prompt="test",
            initial_node="disclaimer",
            nodes=[
                FlowNode(
                    id="disclaimer",
                    name="Disclaimer",
                    instruction=(
                        "Read the disclaimer. "
                        "Proceed immediately — no caller response needed."
                    ),
                    edges=[
                        Edge(
                            id="e_next",
                            condition="disclaimer read",
                            target_node_id="main",
                        ),
                    ],
                ),
                FlowNode(
                    id="main",
                    name="Main",
                    static_text="Welcome!",
                    is_final=True,
                ),
            ],
        )

        machine = await _make_machine(flow)
        machine.mark_node_spoken()
        # No user message — but auto-proceed should skip Gate 4A

        r = await machine.request_transition("e_next")
        assert r.allowed is True

    @pytest.mark.anyio
    async def test_user_turns_reset_on_transition(self) -> None:
        """user_turns_in_node resets to 0 when entering new node."""
        machine = await _make_machine(_simple_flow())

        machine.context.add_user_message("hello")
        assert machine.context.user_turns_in_node == 1

        machine.mark_node_spoken()
        await machine.apply_transition("e_start_mid")

        # After transition, user_turns resets
        assert machine.context.user_turns_in_node == 0
        assert machine.context.user_spoke_in_node is False

    @pytest.mark.anyio
    async def test_multiple_user_messages_tracked(self) -> None:
        """Multiple user messages in one node are all counted."""
        machine = await _make_machine(_simple_flow())

        machine.context.add_user_message("first")
        machine.context.add_user_message("second")
        machine.context.add_user_message("third")

        assert machine.context.user_turns_in_node == 3
        assert machine.context.user_spoke_in_node is True
