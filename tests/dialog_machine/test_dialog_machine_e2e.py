"""End-to-end tests for DialogStateMachine + LiveKit integration.

Covers all layers:
  1. DialogStateMachine core (creation, transitions, context, global edges)
  2. LiveKit bridge (ToolDescriptor → FunctionTool conversion)
  3. FlowNodeTask (global edge interrupt → detour → auto-return)
  4. FlowActionRunner (action execution, speak/generate_reply pass-throughs)
  5. Template rendering ({{ name }}, {{ phone }}, {{ userdata.x }})
  6. SimpleFlowAgent (smoke test, language filtering)

Mocked: LiveKit AgentSession (session.say / session.generate_reply)
Not testable offline: Real LiveKit room, actual TTS/STT
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_FLOW_PATH = str(
    Path(__file__).resolve().parents[4]
    / "super"
    / "core"
    / "voice"
    / "dialog_machine"
    / "testing"
    / "sample_appointment_flow.json"
)


@pytest.fixture
def flow():
    """Load sample appointment flow."""
    from superdialog.flow.models import ConversationFlow

    return ConversationFlow.from_file(SAMPLE_FLOW_PATH)


@pytest.fixture
def mock_adapter():
    """MockAdapter with a full happy-path edge sequence."""
    from superdialog.machine.testing.mock_adapter import MockAdapter

    return MockAdapter(
        edge_sequence=[
            "greeting_to_collect_info",
            "info_to_confirm",
            "confirm_to_goodbye",
        ]
    )


@pytest.fixture
def user_state():
    """Minimal UserState for testing."""
    from super.core.voice.schema import UserState

    return UserState(
        thread_id="test-thread-e2e",
        user_name="Alice Smith",
        contact_number="+1234567890",
        model_config={},
        token="test-token",
    )


@pytest.fixture
def session_state(user_state):
    """SessionState wrapping the test UserState."""
    from super.core.voice.livekit.lite_v2.state import SessionState

    return SessionState(
        session_id="test-session-e2e",
        user_state=user_state,
        callback=None,
        config={
            "handover_enabled": True,
            "knowledge_base": {"name": "test-kb"},
        },
    )


@pytest.fixture
def machine(flow, mock_adapter):
    """Ready-to-use DialogStateMachine at greeting node."""
    from superdialog.machine.machine import DialogStateMachine

    m = asyncio.get_event_loop().run_until_complete(
        DialogStateMachine.from_flow(flow, mock_adapter, session_id="e2e-test")
    )
    return m


@pytest.fixture
def seeded_machine(machine):
    """Machine with userdata seeded (as handler does)."""
    machine.context.userdata.update({"name": "Alice Smith", "phone": "+1234567890"})
    return machine


# ---------------------------------------------------------------------------
# 1. DialogStateMachine Core
# ---------------------------------------------------------------------------


class TestDialogStateMachineCore:
    """Test machine creation, transitions, context tracking."""

    def test_machine_creation(self, machine, flow):
        """Machine initializes at the flow's initial node."""
        assert machine.current_state == flow.initial_node
        assert machine.current_node.id == "greeting"
        assert not machine.current_node.is_final

    def test_tool_descriptors(self, machine):
        """get_tools_for_node returns edge + global descriptors."""
        tools = machine.get_tools_for_node(machine.current_node)
        ids = {t.id for t in tools}
        assert "greeting_to_collect_info" in ids
        assert "greeting_to_faq" in ids
        assert "global_faq" in ids
        assert len(tools) == 3

    @pytest.mark.anyio
    async def test_happy_path_transitions(self, machine, mock_adapter):
        """Walk greeting → collect_info → confirm → goodbye."""
        r1 = await machine.apply_transition(
            "greeting_to_collect_info", user_input="I want to book"
        )
        assert r1.outcome == "transition"
        assert r1.from_node == "greeting"
        assert r1.to_node == "collect_info"
        assert machine.current_state == "collect_info"

        r2 = await machine.apply_transition(
            "info_to_confirm", user_input="John, tomorrow 3pm"
        )
        assert r2.to_node == "confirm"

        r3 = await machine.apply_transition("confirm_to_goodbye", user_input="Yes")
        assert r3.to_node == "goodbye"
        assert machine.current_node.is_final
        assert mock_adapter.session_ended

    @pytest.mark.anyio
    async def test_context_tracking(self, machine):
        """Visit counts and transition log are maintained."""
        await machine.apply_transition("greeting_to_collect_info")
        await machine.apply_transition("info_to_confirm")
        await machine.apply_transition("confirm_to_goodbye")

        assert machine.context.visit_count == {
            "greeting": 1,
            "collect_info": 1,
            "confirm": 1,
            "goodbye": 1,
        }
        log = machine.context.transition_log
        assert len(log) == 3
        assert log[0].from_node == "greeting"
        assert log[2].to_node == "goodbye"

    @pytest.mark.anyio
    async def test_global_edge_pushes_intent_stack(self, flow):
        """Global edge pushes interrupted node onto intent stack."""
        from superdialog.machine.machine import DialogStateMachine
        from superdialog.machine.testing.mock_adapter import MockAdapter

        adapter = MockAdapter(edge_sequence=["global_faq", "faq_to_collect_info"])
        m = await DialogStateMachine.from_flow(flow, adapter, session_id="global-test")

        r = await m.apply_transition("global_faq", user_input="What services?")
        assert r.to_node == "faq"
        assert len(m.context.intent_stack) == 1
        assert m.context.intent_stack[0].node_id == "greeting"

    @pytest.mark.anyio
    async def test_tools_change_per_node(self, machine):
        """Tools reflect the current node's edges, not the previous ones."""
        tools_greeting = {
            t.id for t in machine.get_tools_for_node(machine.current_node)
        }
        assert "greeting_to_collect_info" in tools_greeting

        await machine.apply_transition("greeting_to_collect_info")

        tools_collect = {t.id for t in machine.get_tools_for_node(machine.current_node)}
        assert "info_to_confirm" in tools_collect
        assert "greeting_to_collect_info" not in tools_collect


# ---------------------------------------------------------------------------
# 2. LiveKit Bridge
# ---------------------------------------------------------------------------


class TestLiveKitBridge:
    """Test ToolDescriptor → LiveKit FunctionTool conversion."""

    def test_descriptors_to_livekit_tools(self, machine):
        """descriptors_to_livekit_tools creates FunctionTool objects."""
        from superdialog.machine.adapters.livekit_bridge import (
            descriptors_to_livekit_tools,
        )

        descriptors = machine.get_tools_for_node(machine.current_node)
        tools = descriptors_to_livekit_tools(
            descriptors, machine, new_agent_fn=lambda m: f"Agent({m.current_state})"
        )
        assert len(tools) == 3
        for t in tools:
            assert hasattr(t, "info"), "FunctionTool should have .info"

    def test_tool_names_match_descriptors(self, machine):
        """LiveKit tool names match descriptor IDs."""
        from superdialog.machine.adapters.livekit_bridge import (
            descriptors_to_livekit_tools,
        )

        descriptors = machine.get_tools_for_node(machine.current_node)
        tools = descriptors_to_livekit_tools(
            descriptors, machine, new_agent_fn=lambda m: None
        )
        lk_names = {t.info.name for t in tools}
        desc_ids = {d.id for d in descriptors}
        assert lk_names == desc_ids

    @pytest.mark.anyio
    async def test_new_agent_fn_called_on_transition(self, machine):
        """new_agent_fn is invoked when a transition tool fires."""
        from superdialog.machine.adapters.livekit_bridge import (
            descriptors_to_livekit_tools,
        )

        agents_created = []

        def track_agent(m):
            agents_created.append(m.current_state)
            return f"Agent({m.current_state})"

        descriptors = machine.get_tools_for_node(machine.current_node)
        tools = descriptors_to_livekit_tools(
            descriptors, machine, new_agent_fn=track_agent
        )

        # Find and invoke the greeting_to_collect_info tool
        target_tool = None
        for t in tools:
            if t.info.name == "greeting_to_collect_info":
                target_tool = t
                break
        assert target_tool is not None

        # Invoke the tool's callable directly
        mock_ctx = MagicMock()
        result = await target_tool._callable(mock_ctx)
        assert "collect_info" in agents_created
        assert result == "Agent(collect_info)"


# ---------------------------------------------------------------------------
# 5. FlowNodeTask (global edge detour)
# ---------------------------------------------------------------------------


class TestFlowNodeTask:
    """Test global edge interrupt → detour node → auto-return."""

    def test_task_creation(self, machine):
        """FlowNodeTask is created with correct tools for detour node."""
        from superdialog.machine.adapters.livekit_bridge import get_flow_node_task_class

        # Navigate to FAQ (global edge target)
        asyncio.get_event_loop().run_until_complete(
            machine.apply_transition("global_faq")
        )
        assert machine.current_state == "faq"

        TaskCls = get_flow_node_task_class()
        task = TaskCls(machine=machine, node_id="faq")
        tool_names = {t.info.name for t in task._tools}
        assert "faq_to_collect_info" in tool_names or "faq_to_goodbye" in tool_names

    def test_task_receives_platform_tools(self, machine):
        """FlowNodeTask includes platform tools when passed."""
        from superdialog.machine.adapters.livekit_bridge import get_flow_node_task_class

        # Navigate to FAQ
        asyncio.get_event_loop().run_until_complete(
            machine.apply_transition("global_faq")
        )

        # Create mock platform tools
        mock_end_call = MagicMock()
        mock_end_call.info.name = "end_call"
        mock_get_docs = MagicMock()
        mock_get_docs.info.name = "get_docs"

        TaskCls = get_flow_node_task_class()
        task = TaskCls(
            machine=machine,
            node_id="faq",
            platform_tools=[mock_end_call, mock_get_docs],
        )
        tool_names = {t.info.name for t in task._tools}
        assert "end_call" in tool_names
        assert "get_docs" in tool_names

    @pytest.mark.anyio
    async def test_global_edge_intent_stack_round_trip(self, flow):
        """Global edge → detour → return restores interrupted node."""
        from superdialog.machine.machine import DialogStateMachine
        from superdialog.machine.testing.mock_adapter import MockAdapter

        adapter = MockAdapter(
            edge_sequence=[
                "global_faq",
                "faq_to_collect_info",
                "info_to_confirm",
                "confirm_to_goodbye",
            ]
        )
        m = await DialogStateMachine.from_flow(flow, adapter, session_id="detour-test")

        # Trigger global FAQ from greeting
        r1 = await m.apply_transition("global_faq")
        assert r1.to_node == "faq"
        assert len(m.context.intent_stack) == 1

        # Return from FAQ to collect_info
        r2 = await m.apply_transition("faq_to_collect_info")
        assert r2.to_node == "collect_info"

        # Continue normal flow
        r3 = await m.apply_transition("info_to_confirm")
        assert r3.to_node == "confirm"

        r4 = await m.apply_transition("confirm_to_goodbye")
        assert r4.to_node == "goodbye"
        assert m.current_node.is_final


# ---------------------------------------------------------------------------
# 6. FlowActionRunner (renamed from LiveKitRuntimeAdapter)
# ---------------------------------------------------------------------------


class TestFlowActionRunner:
    """Test the action runner that bridges DialogStateMachine to LiveKit."""

    def test_speak_is_passthrough(self):
        """speak() stores text but does not call TTS."""
        from superdialog.machine.adapters.livekit_adapter import FlowActionRunner

        runner = FlowActionRunner(action_executor=MagicMock())
        node = MagicMock()
        node.id = "test_node"

        asyncio.get_event_loop().run_until_complete(runner.speak("Hello world", node))
        assert runner._pending_response == "Hello world"

    def test_generate_reply_is_passthrough(self):
        """generate_reply() stores instruction but does not call LLM."""
        from superdialog.machine.adapters.livekit_adapter import FlowActionRunner

        runner = FlowActionRunner(action_executor=MagicMock())
        node = MagicMock()
        node.id = "test_node"

        result = asyncio.get_event_loop().run_until_complete(
            runner.generate_reply("Ask the user their name", node)
        )
        assert result == "Ask the user their name"
        assert runner._pending_response == "Ask the user their name"

    def test_end_session_sets_flag(self):
        """end_session() sets session_ended flag."""
        from superdialog.machine.adapters.livekit_adapter import FlowActionRunner

        runner = FlowActionRunner(action_executor=MagicMock())
        assert not runner.session_ended

        asyncio.get_event_loop().run_until_complete(runner.end_session())
        assert runner.session_ended

    @pytest.mark.anyio
    async def test_execute_action_delegates_to_executor(self):
        """execute_action() calls ActionExecutor.execute_action()."""
        from superdialog.machine.adapters.livekit_adapter import FlowActionRunner

        mock_executor = MagicMock()
        mock_executor.execute_action = AsyncMock(
            return_value={"status": "ok", "data": "result"}
        )
        mock_executor.__aenter__ = AsyncMock(return_value=mock_executor)
        mock_executor.__aexit__ = AsyncMock(return_value=False)

        runner = FlowActionRunner(action_executor=mock_executor)

        action = MagicMock()
        action.id = "send_confirmation_sms"

        result = await runner.execute_action(action, userdata={"phone": "+1234567890"})
        assert result == {"status": "ok", "data": "result"}
        mock_executor.execute_action.assert_called_once()
        call_args = mock_executor.execute_action.call_args
        assert call_args[0][0] == "send_confirmation_sms"
        assert hasattr(call_args[0][1], "phone")

    def test_supports_criteria_is_false(self):
        """FlowActionRunner declares supports_criteria=False."""
        from superdialog.machine.adapters.livekit_adapter import FlowActionRunner

        runner = FlowActionRunner(action_executor=MagicMock())
        assert runner.supports_criteria is False

    def test_backward_compat_alias(self):
        """LiveKitRuntimeAdapter alias still works for migration."""
        from superdialog.machine.adapters.livekit_adapter import (
            FlowActionRunner,
            LiveKitRuntimeAdapter,
        )

        assert LiveKitRuntimeAdapter is FlowActionRunner


# ---------------------------------------------------------------------------
# 7. Template Rendering
# ---------------------------------------------------------------------------


class TestTemplateRendering:
    """Test {{ name }}, {{ phone }}, {{ userdata.x }} substitution."""

    def test_render_name_and_phone(self, seeded_machine, session_state):
        """{{ name }} and {{ phone }} resolve from user_state."""
        from superdialog.machine.composer import (
            render_template as _render_flow_template,
        )

        result = _render_flow_template(
            "Hello {{ name }}, calling {{ phone }}",
            seeded_machine,
        )
        assert "Alice Smith" in result
        assert "+1234567890" in result

    def test_render_userdata(self, seeded_machine, session_state):
        """{{ userdata.x }} resolves from machine context."""
        from superdialog.machine.composer import (
            render_template as _render_flow_template,
        )

        seeded_machine.context.userdata["appointment_date"] = "March 20"

        result = _render_flow_template(
            "Your appointment is on {{ userdata.appointment_date }}",
            seeded_machine,
        )
        assert "March 20" in result

    def test_render_plain_text_passthrough(self, seeded_machine, session_state):
        """Text without templates passes through unchanged."""
        from superdialog.machine.composer import (
            render_template as _render_flow_template,
        )

        text = "Hello, how can I help you today?"
        result = _render_flow_template(text, seeded_machine)
        assert result == text


# ---------------------------------------------------------------------------
# 8. Smoke Test (BFS shortest path)
# ---------------------------------------------------------------------------


class TestSmokeTest:
    """Test the built-in smoke test runner."""

    @pytest.mark.anyio
    async def test_smoke_completes(self):
        """Smoke test finds a path to final node."""
        from superdialog.machine.testing.flow_smoke import smoke_test_flow_path_async

        result = await smoke_test_flow_path_async(SAMPLE_FLOW_PATH)
        assert result.is_complete
        assert result.final_state == "goodbye"
        assert result.transitions >= 2


# ---------------------------------------------------------------------------
# 10. Language filtering
# ---------------------------------------------------------------------------


class TestLanguageFiltering:
    """Test _filter_language_markers and _resolve_active_language."""

    def test_filter_en_from_dual_language(self):
        from superdialog.machine.composer import (
            filter_language_markers as _filter_language_markers,
        )

        text = (
            "[EN] Thank you for calling.\n"
            "[HI] कॉल करने के लिए धन्यवाद.\n"
            "\nWait for response."
        )
        result = _filter_language_markers(text, "en")
        assert "Thank you for calling." in result
        assert "कॉल करने" not in result
        assert "Wait for response." in result

    def test_filter_hi_from_dual_language(self):
        from superdialog.machine.composer import (
            filter_language_markers as _filter_language_markers,
        )

        text = (
            "[EN] Thank you for calling.\n"
            "[HI] कॉल करने के लिए धन्यवाद.\n"
            "\nWait for response."
        )
        result = _filter_language_markers(text, "hi")
        assert "कॉल करने के लिए धन्यवाद." in result
        assert "Thank you for calling." not in result
        assert "Wait for response." in result

    def test_no_markers_returns_original(self):
        from superdialog.machine.composer import (
            filter_language_markers as _filter_language_markers,
        )

        text = "Plain text without markers."
        assert _filter_language_markers(text, "en") == text

    def test_fallback_to_en_when_target_missing(self):
        from superdialog.machine.composer import (
            filter_language_markers as _filter_language_markers,
        )

        text = "[EN] English only text.\n[HI] Hindi text."
        result = _filter_language_markers(text, "fr")
        assert "English only text." in result
        assert "Hindi text." not in result

    def test_empty_text(self):
        from superdialog.machine.composer import (
            filter_language_markers as _filter_language_markers,
        )

        assert _filter_language_markers("", "en") == ""

    def test_resolve_language_from_machine_context(self):
        from superdialog.machine.composer import resolve_active_language

        machine = MagicMock()
        machine.context.agent_language = "hi"
        state = MagicMock()
        assert resolve_active_language(state, machine) == "hi"

    def test_resolve_language_from_config(self):
        from superdialog.machine.composer import resolve_active_language

        machine = MagicMock()
        machine.context.agent_language = ""
        machine._flow.agent_language = ""
        state = MagicMock()
        state.config = {"language": "hi"}
        assert resolve_active_language(state, machine) == "hi"

    def test_resolve_language_default_en(self):
        from superdialog.machine.composer import resolve_active_language

        machine = MagicMock()
        machine.context.agent_language = ""
        machine._flow.agent_language = ""
        state = MagicMock()
        state.config = {}
        assert resolve_active_language(state, machine) == "en"
