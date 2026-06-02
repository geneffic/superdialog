"""Tests for SimpleFlowAgent — DialogStateMachine + LiveKit Agent integration.

Tests cover:
- DialogStateMachine integration (flow creation, tools, transitions)
- Enriched instructions (routing context, edge summaries, slot data)
- Tool guard (prevents LLM tool hallucination)
- Initial node suppression (prevents realtime model auto-greeting)
- Template rendering in speech (process_text resolves variables)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from superdialog.flow.models import ConversationFlow, Edge, FlowNode
from superdialog.machine.machine import DialogStateMachine

# ------------------------------------------------------------------
# Flow fixtures
# ------------------------------------------------------------------


def _make_two_node_flow() -> ConversationFlow:
    """Greeting → Farewell (final)."""
    return ConversationFlow(
        system_prompt="You are a helpful assistant.",
        initial_node="greeting",
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                instruction="Greet the user warmly.",
                edges=[
                    Edge(
                        id="proceed_to_farewell",
                        condition="User is ready to end the call",
                        target_node_id="farewell",
                    ),
                ],
            ),
            FlowNode(
                id="farewell",
                name="Farewell",
                static_text="Goodbye! Have a great day.",
                is_final=True,
            ),
        ],
    )


def _make_three_node_flow() -> ConversationFlow:
    """Greeting → Collect Name → Farewell."""
    return ConversationFlow(
        system_prompt="You are an intake agent.",
        initial_node="greeting",
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                instruction="Greet the user and ask for their name.",
                edges=[
                    Edge(
                        id="collect_name",
                        condition="User provides their name",
                        target_node_id="collect",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The user's name",
                                },
                            },
                            "required": ["name"],
                        },
                    ),
                ],
            ),
            FlowNode(
                id="collect",
                name="Confirm Name",
                instruction="Confirm the user's name: {{ name }}",
                edges=[
                    Edge(
                        id="finish",
                        condition="User confirms",
                        target_node_id="farewell",
                    ),
                ],
            ),
            FlowNode(
                id="farewell",
                name="Farewell",
                static_text="Thank you, goodbye!",
                is_final=True,
            ),
        ],
    )


def _make_template_flow() -> ConversationFlow:
    """Flow with {{ name }} template."""
    return ConversationFlow(
        system_prompt="You are a welcome agent.",
        initial_node="greet",
        nodes=[
            FlowNode(
                id="greet",
                name="Greet",
                static_text="Hello {{ name }}, welcome!",
                edges=[],
                is_final=True,
            ),
        ],
    )


class _StubAdapter:
    """Minimal adapter for tests — all no-ops."""

    async def speak(self, text, node):
        pass

    async def generate_reply(self, instruction, node, history=None, userdata=None):
        return instruction

    async def evaluate_criteria(self, node, history, userdata):
        raise NotImplementedError

    async def execute_action(self, action, userdata):
        return None

    async def generate_recovery(self, node, error):
        return "Sorry"

    async def end_session(self):
        pass


# ------------------------------------------------------------------
# Helpers for creating SimpleFlowAgent with mocked LiveKit
# ------------------------------------------------------------------


def _make_stub_state() -> MagicMock:
    """Create a minimal mock SessionState for tests."""
    state = MagicMock()
    state.session_id = "test-session"
    state.config = {}
    state.is_shutting_down = False
    state.mark_processing = MagicMock()
    return state


def _create_agent(machine, state=None, config=None):
    """Create a SimpleFlowAgent with scope and mocked LiveKit deps."""
    from superdialog.machine.adapters.simple_agent import SimpleFlowAgent

    if state is None:
        state = _make_stub_state()

    scope = machine.build_node_scope()

    with (
        patch(
            "super.core.voice.dialog_machine.adapters.simple_agent"
            ".descriptors_to_livekit_tools",
            return_value=[],
        ),
        patch("super.core.voice.tools.PlatformToolkit") as mock_toolkit,
    ):
        mock_toolkit.return_value.build_tools.return_value = []
        agent = SimpleFlowAgent(
            machine=machine, state=state, config=config, scope=scope
        )
    return agent


# ------------------------------------------------------------------
# DialogStateMachine integration tests
# ------------------------------------------------------------------


class TestDialogMachineIntegration:
    """Test that DialogStateMachine works correctly with our flow definitions."""

    @pytest.mark.anyio
    async def test_two_node_flow_creation(self) -> None:
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-1"
        )
        assert machine.current_state == "greeting"
        assert not machine.is_complete

    @pytest.mark.anyio
    async def test_two_node_tools(self) -> None:
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-2"
        )
        tools = machine.get_tools_for_node()
        assert len(tools) == 1
        assert tools[0].id == "proceed_to_farewell"
        assert not tools[0].is_data_collection

    @pytest.mark.anyio
    async def test_two_node_transition(self) -> None:
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-3"
        )
        result = await machine.apply_transition("proceed_to_farewell")
        assert result.outcome == "transition"
        assert result.from_node == "greeting"
        assert result.to_node == "farewell"
        assert machine.current_state == "farewell"
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_three_node_data_collection(self) -> None:
        flow = _make_three_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-4"
        )
        tools = machine.get_tools_for_node()
        assert len(tools) == 1
        assert tools[0].id == "collect_name"
        assert tools[0].is_data_collection
        assert tools[0].input_schema is not None

        result = await machine.apply_transition(
            "collect_name", collected_data={"name": "Alice"}
        )
        assert result.outcome == "transition"
        assert machine.current_state == "collect"
        assert machine.context.node_slots.get("greeting", {}).get("name") == "Alice"

    @pytest.mark.anyio
    async def test_enriched_instructions(self) -> None:
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-5"
        )
        instructions = machine.get_enriched_instructions()
        assert "Greet the user warmly" in instructions
        assert "proceed_to_farewell" in instructions

    @pytest.mark.anyio
    async def test_final_node_has_no_tools(self) -> None:
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-6"
        )
        await machine.apply_transition("proceed_to_farewell")
        tools = machine.get_tools_for_node()
        assert len(tools) == 0


# ------------------------------------------------------------------
# SimpleFlowAgent enriched instructions tests
# ------------------------------------------------------------------


class TestEnrichedInstructions:
    """Test that SimpleFlowAgent builds enriched instructions."""

    @pytest.mark.anyio
    async def test_instructions_contain_system_prompt(self) -> None:
        """System prompt should be included in agent instructions."""
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-enrich-1"
        )
        agent = _create_agent(machine)
        assert "helpful assistant" in agent._instructions

    @pytest.mark.anyio
    async def test_scope_node_instruction_contains_node_text(self) -> None:
        """Node instruction text should appear in scope node_instruction."""
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-enrich-2"
        )
        agent = _create_agent(machine)
        assert "Greet the user warmly" in agent._scope.node_instruction

    @pytest.mark.anyio
    async def test_scope_node_instruction_contains_edge_summaries(self) -> None:
        """Edge transition summaries should appear in scope node_instruction."""
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-enrich-3"
        )
        agent = _create_agent(machine)
        assert "proceed_to_farewell" in agent._scope.node_instruction

    @pytest.mark.anyio
    async def test_scope_slot_data_after_collection(
        self,
    ) -> None:
        """Collected slot data in scope and userdata available for templates."""
        flow = _make_three_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-enrich-4"
        )
        # Transition with data to "collect" node
        await machine.apply_transition("collect_name", collected_data={"name": "Alice"})
        agent = _create_agent(machine)
        # The scope node_instruction should mention name
        assert "name" in agent._scope.node_instruction
        # Userdata has the collected value (used by process_text at
        # speech time)
        assert machine.context.userdata["name"] == "Alice"

    @pytest.mark.anyio
    async def test_scope_node_instruction_completed_nodes_context(
        self,
    ) -> None:
        """Mid-conversation context should mention completed nodes."""
        flow = _make_three_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-enrich-5"
        )
        await machine.apply_transition("collect_name", collected_data={"name": "Bob"})
        agent = _create_agent(machine)
        # Should mention "greeting" as completed in scope instruction
        assert "greeting" in agent._scope.node_instruction.lower()
        assert "mid-conversation" in agent._scope.node_instruction.lower()


# ------------------------------------------------------------------
# Tool guard tests
# ------------------------------------------------------------------


class TestToolGuard:
    """Test tool guard directive in instructions."""

    @pytest.mark.anyio
    async def test_tool_guard_present_with_tools(self) -> None:
        """When tools exist, tool guard directive should be appended."""
        from superdialog.machine.adapters.simple_agent import SimpleFlowAgent

        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-guard-1"
        )

        # Create a mock tool with info.name
        mock_tool = MagicMock()
        mock_tool.info.name = "proceed_to_farewell"

        state = _make_stub_state()
        scope = machine.build_node_scope()
        # Single agent always has edge tools
        with (
            patch(
                "super.core.voice.dialog_machine.adapters.simple_agent"
                ".descriptors_to_livekit_tools",
                return_value=[mock_tool],
            ),
            patch("super.core.voice.tools.PlatformToolkit") as mock_tk,
        ):
            mock_tk.return_value.build_tools.return_value = []
            agent = SimpleFlowAgent(
                machine=machine,
                state=state,
                scope=scope,
            )

        assert "You ONLY have these tools" in agent._instructions
        assert "proceed_to_farewell" in agent._instructions
        assert "NEVER call any tool not in this list" in agent._instructions

    @pytest.mark.anyio
    async def test_no_tool_guard_without_tools(self) -> None:
        """When no tools have info, tool guard should not be appended."""
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-guard-2"
        )
        # _create_agent uses empty tool lists
        agent = _create_agent(machine)
        assert "You ONLY have these tools" not in agent._instructions


# ------------------------------------------------------------------
# Initial node suppression tests
# ------------------------------------------------------------------


class TestInitialNodeSuppression:
    """Test initial node auto-greeting suppression."""

    @pytest.mark.anyio
    async def test_initial_node_has_suppression(self) -> None:
        """Initial node should have auto-greeting suppression."""
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-init-1"
        )
        agent = _create_agent(machine)
        assert agent._is_initial_node is True
        assert "Do NOT generate any speech" in agent._instructions

    @pytest.mark.anyio
    async def test_non_initial_node_no_suppression(self) -> None:
        """Non-initial nodes should NOT have suppression."""
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-init-2"
        )
        await machine.apply_transition("proceed_to_farewell")
        agent = _create_agent(machine)
        assert agent._is_initial_node is False
        assert "Do NOT generate any speech" not in agent._instructions


# ------------------------------------------------------------------
# Template rendering in speech tests
# ------------------------------------------------------------------


class TestSpeechTemplateRendering:
    """Test that _speak_node renders template variables."""

    @pytest.mark.anyio
    async def test_process_text_renders_userdata_templates(self) -> None:
        """process_text should resolve {{ name }} from machine userdata."""
        from superdialog.machine.composer import process_text

        flow = _make_template_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-speech-1"
        )
        machine.context.userdata["name"] = "Alice"

        result = process_text("Hello {{ name }}, welcome!", machine, "en")
        assert "Alice" in result
        assert "{{ name }}" not in result

    @pytest.mark.anyio
    async def test_process_text_no_template_passthrough(self) -> None:
        """Text without templates should pass through unchanged."""
        from superdialog.machine.composer import process_text

        flow = _make_template_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-speech-2"
        )
        result = process_text("No templates here", machine, "en")
        assert result == "No templates here"

    @pytest.mark.anyio
    async def test_process_text_undefined_vars_render_empty(self) -> None:
        """Undefined template variables should render as empty."""
        from superdialog.machine.composer import process_text

        flow = _make_template_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-speech-3"
        )
        machine.context.userdata["name"] = "Bob"

        result = process_text(
            "Hello {{ name }}, code: {{ unknown_var }}", machine, "en"
        )
        assert "Bob" in result
        assert "{{ name }}" not in result
        assert "{{ unknown_var }}" not in result

    @pytest.mark.anyio
    async def test_process_text_language_filtering(self) -> None:
        """Language markers should be filtered correctly."""
        from superdialog.machine.composer import process_text

        flow = ConversationFlow(
            system_prompt="Test",
            initial_node="greet",
            agent_language="hi",
            nodes=[
                FlowNode(
                    id="greet",
                    name="Greet",
                    static_text="[EN] Hello\n[HI] नमस्ते",
                    edges=[],
                    is_final=True,
                ),
            ],
        )
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-speech-4"
        )

        result = process_text("[EN] Hello\n[HI] नमस्ते", machine, "hi")
        assert "नमस्ते" in result
        assert "Hello" not in result

    @pytest.mark.anyio
    async def test_process_text_action_results_in_templates(self) -> None:
        """Action results should be available in templates."""
        from superdialog.machine.composer import process_text

        flow = _make_template_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-speech-5"
        )
        machine.context.action_results["api_call"] = {"status": "success"}

        result = process_text("Result: {{ actions.api_call.status }}", machine, "en")
        assert "success" in result


# ------------------------------------------------------------------
# Scope-based initialization tests
# ------------------------------------------------------------------


class TestScopeBasedInit:
    """Test scope-based initialization of SimpleFlowAgent."""

    @pytest.mark.anyio
    async def test_scope_required(self) -> None:
        """Omitting scope raises TypeError."""
        from superdialog.machine.adapters.simple_agent import SimpleFlowAgent

        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-scope-req"
        )
        state = _make_stub_state()
        with pytest.raises(TypeError):
            SimpleFlowAgent(machine=machine, state=state)

    @pytest.mark.anyio
    async def test_scope_sets_language(self) -> None:
        """Language from scope is used."""
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-scope-lang"
        )
        agent = _create_agent(machine)
        assert agent._active_lang == "en"

    @pytest.mark.anyio
    async def test_scope_sets_auto_proceed(self) -> None:
        """auto_proceed from scope is stored."""
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-scope-auto"
        )
        agent = _create_agent(machine)
        assert agent._auto_proceed is False

    @pytest.mark.anyio
    async def test_scope_sets_initial_node(self) -> None:
        """is_initial from scope is stored."""
        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-scope-init"
        )
        agent = _create_agent(machine)
        assert agent._is_initial_node is True


# ------------------------------------------------------------------
# FlowExecutor requirement tests
# ------------------------------------------------------------------


class TestFlowExecutorRequired:
    """Test that transitions require FlowExecutor."""

    @pytest.mark.anyio
    async def test_transition_without_executor_raises(self) -> None:
        """_gated_transition raises RuntimeError if no executor."""
        from superdialog.machine.models import TurnResult

        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-exec-req"
        )
        agent = _create_agent(machine)

        turn_result = TurnResult(
            outcome="transition",
            from_node="greeting",
            to_node="farewell",
            response="ok",
            edge_id="proceed_to_farewell",
        )

        with pytest.raises(RuntimeError, match="FlowExecutor"):
            await agent._gated_transition("proceed_to_farewell", turn_result)

    @pytest.mark.anyio
    async def test_transition_with_executor_delegates(self) -> None:
        """_gated_transition delegates to executor.handle_transition."""
        from unittest.mock import AsyncMock

        from superdialog.machine.models import TurnResult

        flow = _make_two_node_flow()
        machine = await DialogStateMachine.from_flow(
            flow, _StubAdapter(), session_id="test-exec-del"
        )
        agent = _create_agent(machine)

        mock_executor = MagicMock()
        mock_executor.handle_transition = AsyncMock(return_value=None)
        agent._executor = mock_executor

        turn_result = TurnResult(
            outcome="transition",
            from_node="greeting",
            to_node="farewell",
            response="ok",
            edge_id="proceed_to_farewell",
        )

        await agent._gated_transition("proceed_to_farewell", turn_result)
        mock_executor.handle_transition.assert_called_once_with(
            "proceed_to_farewell", turn_result, agent
        )
