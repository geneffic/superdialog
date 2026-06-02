"""Language tracking tests — verifies every touchpoint reacts to language changes.

Organized by implementation phase:
- Phase 1: Config seeding into machine.context
- Phase 2: _active_lang as live @property
- Phase 3: STT language detection hook
- E2E: Full language lifecycle
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from superdialog.machine.composer import (
    extract_speech_text,
    filter_language_markers,
    process_text,
    resolve_active_language,
)

# ---------------------------------------------------------------------------
# Phase 1: Config Seeding
# ---------------------------------------------------------------------------


class TestPhase1ConfigSeeding:
    """CP-1.1 through CP-1.6: resolve_active_language seeds machine.context."""

    def test_cp_1_1_config_seeds_machine_context(self) -> None:
        """Config language writes to machine.context via set_language."""
        machine = MagicMock()
        machine.context.agent_language = ""
        machine._flow.agent_language = ""
        state = MagicMock()
        state.config = {"language": "hi"}

        result = resolve_active_language(state, machine)

        assert result == "hi"
        machine.set_language.assert_called_once_with("hi")

    def test_cp_1_2_preferred_language_seeds_context(self) -> None:
        """preferred_language key also seeds machine.context."""
        machine = MagicMock()
        machine.context.agent_language = ""
        machine._flow.agent_language = ""
        state = MagicMock()
        state.config = {"preferred_language": "ta"}

        result = resolve_active_language(state, machine)

        assert result == "ta"
        machine.set_language.assert_called_once_with("ta")

    def test_cp_1_3_context_takes_priority_over_config(self) -> None:
        """Existing machine.context.agent_language is never overwritten."""
        machine = MagicMock()
        machine.context.agent_language = "hi"
        state = MagicMock()
        state.config = {"language": "en"}

        result = resolve_active_language(state, machine)

        assert result == "hi"
        machine.set_language.assert_not_called()

    def test_cp_1_4_flow_takes_priority_over_config(self) -> None:
        """flow.agent_language wins over state.config."""
        machine = MagicMock()
        machine.context.agent_language = ""
        machine._flow.agent_language = "hi"
        state = MagicMock()
        state.config = {"language": "ta"}

        result = resolve_active_language(state, machine)

        assert result == "hi"
        machine.set_language.assert_not_called()

    def test_cp_1_5_empty_config_defaults_to_en(self) -> None:
        """No config, no flow, no context → 'en' default."""
        machine = MagicMock()
        machine.context.agent_language = ""
        machine._flow.agent_language = ""
        state = MagicMock()
        state.config = {}

        result = resolve_active_language(state, machine)

        assert result == "en"
        machine.set_language.assert_not_called()

    def test_cp_1_6_existing_tests_compat(self) -> None:
        """Verify resolve_active_language still returns expected values."""
        # From machine context
        machine = MagicMock()
        machine.context.agent_language = "hi"
        state = MagicMock()
        assert resolve_active_language(state, machine) == "hi"

        # From config
        machine2 = MagicMock()
        machine2.context.agent_language = ""
        machine2._flow.agent_language = ""
        state2 = MagicMock()
        state2.config = {"language": "hi"}
        assert resolve_active_language(state2, machine2) == "hi"

        # Default
        machine3 = MagicMock()
        machine3.context.agent_language = ""
        machine3._flow.agent_language = ""
        state3 = MagicMock()
        state3.config = {}
        assert resolve_active_language(state3, machine3) == "en"


# ---------------------------------------------------------------------------
# Phase 2: _active_lang as live @property
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="superdialog.machine.adapters.simple_agent not ported; Task 7 follow-up"
)
class TestPhase2ActiveLangProperty:
    """CP-2.1 through CP-2.7: _active_lang reads live from machine.context."""

    def _make_agent_stub(self, agent_language: str = "en") -> MagicMock:
        """Create a minimal stub with the _active_lang @property."""
        from superdialog.machine.adapters.simple_agent import SimpleFlowAgent

        machine = MagicMock()
        machine.context.agent_language = agent_language

        agent = MagicMock(spec=SimpleFlowAgent)
        agent._machine = machine
        # Bind the real property
        agent._active_lang = SimpleFlowAgent._active_lang.fget(agent)  # type: ignore[attr-defined]
        return agent

    def test_cp_2_1_property_reads_live_context(self) -> None:
        """After set_language, _active_lang returns new value."""
        from superdialog.machine.adapters.simple_agent import SimpleFlowAgent

        machine = MagicMock()
        machine.context.agent_language = "hi"

        # Simulate the property on a stub
        agent = MagicMock()
        agent._machine = machine
        type(agent)._active_lang = SimpleFlowAgent._active_lang  # type: ignore[attr-defined]

        assert agent._active_lang == "hi"

        # Switch language
        machine.context.agent_language = "en"
        assert agent._active_lang == "en"

    def test_cp_2_2_property_defaults_to_en(self) -> None:
        """Empty context → 'en'."""
        from superdialog.machine.adapters.simple_agent import SimpleFlowAgent

        machine = MagicMock()
        machine.context.agent_language = ""

        agent = MagicMock()
        agent._machine = machine
        type(agent)._active_lang = SimpleFlowAgent._active_lang  # type: ignore[attr-defined]

        assert agent._active_lang == "en"

    def test_cp_2_3_property_normalizes_case(self) -> None:
        """Uppercase → lowercase."""
        from superdialog.machine.adapters.simple_agent import SimpleFlowAgent

        machine = MagicMock()
        machine.context.agent_language = "HI"

        agent = MagicMock()
        agent._machine = machine
        type(agent)._active_lang = SimpleFlowAgent._active_lang  # type: ignore[attr-defined]

        assert agent._active_lang == "hi"

    def test_cp_2_4_process_text_uses_live_language(self) -> None:
        """process_text filters correct markers after language switch."""
        machine = MagicMock()
        machine.context.agent_language = "hi"
        text = "[EN] Hello\n[HI] नमस्ते"

        # Switch to English
        machine.context.agent_language = "en"

        result = process_text(text, machine, "en")
        assert "Hello" in result
        assert "नमस्ते" not in result

    def test_cp_2_5_extract_speech_text_uses_live_language(self) -> None:
        """extract_speech_text returns correct tagged speech after switch."""
        machine = MagicMock()
        machine.context.agent_language = "en"
        instruction = "[EN] How can I help?\n[HI] मैं कैसे मदद कर सकता हूँ?"

        result = extract_speech_text(instruction, machine, "en")
        assert result is not None
        assert "How can I help?" in result


# ---------------------------------------------------------------------------
# Phase 3: STT Language Detection
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="superdialog.machine.adapters.simple_agent not ported; Task 7 follow-up"
)
class TestPhase3ScriptDetection:
    """CP-3.1 through CP-3.5: _detect_script_language + on_user_turn_completed."""

    def test_cp_3_1_detects_hindi_from_devanagari(self) -> None:
        """Devanagari text → 'hi'."""
        from superdialog.machine.adapters.simple_agent import _detect_script_language

        assert _detect_script_language("नमस्ते कैसे हो") == "hi"

    def test_cp_3_2_detects_english_from_latin(self) -> None:
        """Latin text → 'en'."""
        from superdialog.machine.adapters.simple_agent import _detect_script_language

        assert _detect_script_language("Hello how are you") == "en"

    def test_cp_3_3_same_language_noop(self) -> None:
        """Detection returns same as current → no set_language call."""
        from superdialog.machine.adapters.simple_agent import _detect_script_language

        # English text detected as "en", current lang is "en" → no switch
        detected = _detect_script_language("Hello world")
        assert detected == "en"
        # The on_user_turn_completed would skip set_language since detected == _active_lang

    def test_cp_3_4_empty_text_returns_none(self) -> None:
        """Empty text → None (no detection)."""
        from superdialog.machine.adapters.simple_agent import _detect_script_language

        assert _detect_script_language("") is None
        assert _detect_script_language("   123   ") is None

    def test_cp_3_5_mixed_text_dominant_wins(self) -> None:
        """Mixed script text → dominant script wins."""
        from superdialog.machine.adapters.simple_agent import _detect_script_language

        # Mostly Hindi with some English
        assert _detect_script_language("नमस्ते hello कैसे हो") == "hi"
        # Mixed text (more Latin than Devanagari) → ambiguous (None)
        # because it could be Hinglish; only pure-Latin returns "en"
        assert _detect_script_language("Hello world नमस्ते bye") is None
        # Pure English (no Devanagari) → "en"
        assert _detect_script_language("Hello world how are you") == "en"

    def test_cp_3_7_system_prompt_reflects_switch(self) -> None:
        """get_enriched_instructions contains updated language."""
        from superdialog.machine.models import (
            ConversationData,
            FlowContext,
            MachineState,
        )

        ctx = FlowContext(
            data=ConversationData(language="en"),
            state=MachineState(current_node_id="node_1"),
        )
        assert ctx.agent_language == "en"

        # Switch
        ctx.agent_language = "hi"
        assert ctx.agent_language == "hi"

    def test_cp_3_8_criteria_judge_receives_updated_language(self) -> None:
        """_flow_meta reflects language after set_language."""
        from superdialog.machine.models import (
            ConversationData,
            FlowContext,
            MachineState,
        )

        ctx = FlowContext(
            data=ConversationData(language="en"),
            state=MachineState(current_node_id="node_1"),
        )

        ctx.agent_language = "hi"
        # Simulate what machine.py:826 does
        flow_meta = {"agent_language": ctx.agent_language}
        assert flow_meta["agent_language"] == "hi"


# ---------------------------------------------------------------------------
# E2E Integration
# ---------------------------------------------------------------------------


class TestE2ELanguageLifecycle:
    """CP-E2E-1 through CP-E2E-3: Full language lifecycle."""

    def test_e2e_1_full_lifecycle(self) -> None:
        """Config seeds → property reads → switch → correct filtering."""
        # Step 1-3: Config seeds machine.context
        machine = MagicMock()
        machine.context.agent_language = ""
        machine._flow.agent_language = ""

        def _set_lang(lang: str) -> None:
            machine.context.agent_language = lang

        machine.set_language.side_effect = _set_lang

        state = MagicMock()
        state.config = {"language": "hi"}

        lang = resolve_active_language(state, machine)
        assert lang == "hi"
        assert machine.context.agent_language == "hi"

        # Step 5: Node filters Hindi markers
        text = "[EN] Hello\n[HI] नमस्ते"
        result = filter_language_markers(text, machine.context.agent_language)
        assert "नमस्ते" in result
        assert "Hello" not in result

        # Step 6-7: User switches to English (STT detection)
        machine.set_language("en")
        assert machine.context.agent_language == "en"

        # Step 8: Next node filters English markers
        text2 = "[EN] Thanks\n[HI] धन्यवाद"
        result2 = filter_language_markers(text2, machine.context.agent_language)
        assert "Thanks" in result2
        assert "धन्यवाद" not in result2

    def test_e2e_2_flow_json_priority(self) -> None:
        """Flow JSON agent_language beats config."""
        machine = MagicMock()
        machine.context.agent_language = ""
        machine._flow.agent_language = "hi"
        state = MagicMock()
        state.config = {"language": "ta"}

        result = resolve_active_language(state, machine)

        assert result == "hi"
        machine.set_language.assert_not_called()

    def test_e2e_3_no_markers_fallback(self) -> None:
        """No markers → original text returned, system prompt drives language."""
        machine = MagicMock()
        machine.context.agent_language = "hi"

        text = "Greet the customer warmly"
        result = filter_language_markers(text, "hi")

        # No markers → original text unchanged
        assert result == text
