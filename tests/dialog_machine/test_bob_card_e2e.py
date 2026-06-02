"""E2E simulation of the BOB CARD flow through DialogStateMachine.

Walks the happy path using apply_transition() (tool-call mode),
printing progress at each node: node name, speech/instruction,
available tools, and language-filtered text.

Also tests an alternate path (YOB mismatch → security fail).
"""

import logging
from pathlib import Path
from typing import Any

import pytest

from superdialog.flow.models import ConversationFlow
from superdialog.machine.composer import (
    filter_language_markers as _filter_language_markers,
)
from superdialog.machine.machine import DialogStateMachine
from superdialog.machine.models import TurnResult
from superdialog.machine.testing.mock_adapter import MockAdapter

BOB_CARD_FLOW_PATH = str(
    Path(__file__).resolve().parents[4] / "temp" / "flow-bob-card-latest.json"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_node_status(
    machine: DialogStateMachine,
    result: TurnResult | None = None,
    lang: str = "en",
) -> None:
    """Print a formatted status block for the current node."""
    node = machine.current_node
    tools = machine.get_tools_for_node(node)
    tool_ids = [t.id for t in tools]

    # Language-filtered speech
    speech = node.static_text or node.instruction or ""
    filtered = _filter_language_markers(speech, lang) if speech else ""
    # Truncate for readability
    preview = filtered[:200] + "..." if len(filtered) > 200 else filtered

    transition_info = ""
    if result:
        transition_info = (
            f"  edge: {result.edge_id}\n"
            f"  from: {result.from_node} → to: {result.to_node}\n"
        )

    print(
        f"\n{'=' * 60}\n"
        f"NODE: {node.id} ({node.name})\n"
        f"  is_final: {node.is_final}\n"
        f"  type: {'static_text' if node.static_text else 'instruction'}\n"
        f"{transition_info}"
        f"  tools: {tool_ids}\n"
        f"  speech[{lang}]: {preview}\n"
        f"  completed_nodes: {list(machine.context.completed_nodes)}\n"
        f"{'=' * 60}"
    )


async def _build_machine(
    lang: str = "en",
) -> DialogStateMachine:
    """Load BOB CARD flow and create a DialogStateMachine."""
    flow = ConversationFlow.from_file(BOB_CARD_FLOW_PATH)
    adapter = MockAdapter(edge_sequence=[])
    machine = await DialogStateMachine.from_flow(
        flow, adapter, session_id="bob-card-e2e"
    )
    if lang:
        machine.context.agent_language = lang
    return machine


async def _step(
    machine: DialogStateMachine,
    edge_id: str,
    lang: str = "en",
    collected_data: dict[str, Any] | None = None,
) -> TurnResult:
    """Apply a transition and print progress."""
    result = await machine.apply_transition(edge_id, collected_data=collected_data)
    _print_node_status(machine, result, lang)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bob_card_happy_path_en(capfd: Any) -> None:
    """Walk the full happy path in English, printing progress."""
    lang = "en"
    machine = await _build_machine(lang)

    # Starting node
    print("\n\n========== BOB CARD E2E — HAPPY PATH (EN) ==========")
    _print_node_status(machine, lang=lang)

    # Step 1: opening → good_time_check
    await _step(machine, "op_yes", lang)

    # Step 2: good_time_check → disclaimer
    await _step(machine, "gtc_yes", lang)

    # Step 3: disclaimer → yob_ask
    await _step(machine, "disc_proceed", lang)

    # Step 4: yob_ask → yob_echo
    await _step(machine, "yob_given", lang)

    # Step 5: yob_echo → last4_ask (YOB matches 1980)
    await _step(machine, "yob_echo_match", lang)

    # Step 6: last4_ask → last4_echo
    await _step(machine, "l4_given", lang)

    # Step 7: last4_echo → card_received_check
    await _step(machine, "l4_echo_match", lang)

    # Step 8: card_received_check → benefits_offer
    await _step(machine, "crc_yes", lang)

    # Step 9: benefits_offer → benefits_details
    await _step(machine, "bo_yes", lang)

    # Step 10: benefits_details → activation_prompt
    await _step(machine, "bd_proceed", lang)

    # Step 11: activation_prompt → app_check
    await _step(machine, "ap_yes", lang)

    # Step 12: app_check → app_activation_guide
    await _step(machine, "ac_yes", lang)

    # Step 13: app_activation_guide → activation_confirmed
    await _step(machine, "aag_done", lang)

    # Step 14: activation_confirmed → another_query_check
    await _step(machine, "aconf_ack", lang)

    # Step 15: another_query_check → closing (final)
    result = await _step(machine, "aqc_no", lang)

    # Assertions
    assert machine.is_complete
    assert machine.current_state == "closing"
    assert result.outcome == "transition"
    assert len(machine.context.transition_log) == 15

    # Verify all expected nodes were visited
    expected_visited = {
        "opening",
        "good_time_check",
        "disclaimer",
        "yob_ask",
        "yob_echo",
        "last4_ask",
        "last4_echo",
        "card_received_check",
        "benefits_offer",
        "benefits_details",
        "activation_prompt",
        "app_check",
        "app_activation_guide",
        "activation_confirmed",
        "another_query_check",
    }
    assert expected_visited == machine.context.completed_nodes

    print("\n\n✅ Happy path completed: 15 transitions, all nodes visited.")


@pytest.mark.anyio
async def test_bob_card_happy_path_hi(capfd: Any) -> None:
    """Walk the happy path in Hindi, verifying language filtering."""
    lang = "hi"
    machine = await _build_machine(lang)

    print("\n\n========== BOB CARD E2E — HAPPY PATH (HI) ==========")
    _print_node_status(machine, lang=lang)

    # Walk the same path but in Hindi
    await _step(machine, "op_yes", lang)
    await _step(machine, "gtc_yes", lang)
    await _step(machine, "disc_proceed", lang)
    await _step(machine, "yob_given", lang)
    await _step(machine, "yob_echo_match", lang)
    await _step(machine, "l4_given", lang)
    await _step(machine, "l4_echo_match", lang)
    await _step(machine, "crc_yes", lang)
    await _step(machine, "bo_yes", lang)
    await _step(machine, "bd_proceed", lang)
    await _step(machine, "ap_yes", lang)
    await _step(machine, "ac_yes", lang)
    await _step(machine, "aag_done", lang)
    await _step(machine, "aconf_ack", lang)
    result = await _step(machine, "aqc_no", lang)

    assert machine.is_complete
    assert machine.current_state == "closing"

    # Verify Hindi was used in filtered speech
    node = machine._node_map["good_time_check"]
    hi_text = _filter_language_markers(node.instruction or "", "hi")
    assert "credit card" in hi_text.lower() or "welcome call" in hi_text.lower()
    # EN markers should be stripped
    assert "[EN]" not in hi_text
    assert "[HI]" not in hi_text

    print("\n\n✅ Hindi happy path completed.")


@pytest.mark.anyio
async def test_bob_card_yob_mismatch_path() -> None:
    """Test the YOB mismatch → security fail path."""
    lang = "en"
    machine = await _build_machine(lang)

    print("\n\n========== BOB CARD E2E — YOB MISMATCH ==========")
    _print_node_status(machine, lang=lang)

    await _step(machine, "op_yes", lang)
    await _step(machine, "gtc_yes", lang)
    await _step(machine, "disc_proceed", lang)
    await _step(machine, "yob_given", lang)

    # YOB doesn't match → security fail
    result = await _step(machine, "yob_echo_mismatch", lang)

    assert machine.is_complete
    assert machine.current_state == "yob_mismatch_end"
    assert result.outcome == "transition"

    # Final node static_text should be the mismatch message
    node = machine.current_node
    assert node.is_final
    assert "does not match" in (node.static_text or "")

    print("\n\n✅ YOB mismatch path completed — session ended at security fail.")


@pytest.mark.anyio
async def test_bob_card_busy_callback_path() -> None:
    """Test the busy → callback → closing path."""
    lang = "en"
    machine = await _build_machine(lang)

    print("\n\n========== BOB CARD E2E — BUSY CALLBACK ==========")
    _print_node_status(machine, lang=lang)

    await _step(machine, "op_yes", lang)
    # Caller is busy
    await _step(machine, "gtc_busy", lang)
    # Caller agrees to callback
    await _step(machine, "bco_yes", lang)
    # Caller gives exact time
    await _step(machine, "bat_exact", lang)
    # Callback confirmed → closing
    result = await _step(machine, "cc_done", lang)

    assert machine.is_complete
    assert machine.current_state == "closing"

    print("\n\n✅ Busy callback path completed.")


@pytest.mark.anyio
async def test_bob_card_card_not_received_path() -> None:
    """Test card not received → delivery query → closing."""
    lang = "en"
    machine = await _build_machine(lang)

    print("\n\n========== BOB CARD E2E — CARD NOT RECEIVED ==========")
    _print_node_status(machine, lang=lang)

    # Get through security
    await _step(machine, "op_yes", lang)
    await _step(machine, "gtc_yes", lang)
    await _step(machine, "disc_proceed", lang)
    await _step(machine, "yob_given", lang)
    await _step(machine, "yob_echo_match", lang)
    await _step(machine, "l4_given", lang)
    await _step(machine, "l4_echo_match", lang)

    # Card not received
    await _step(machine, "crc_no", lang)
    # Still not found
    await _step(machine, "cnrs_still_no", lang)
    # Yes, raise query
    await _step(machine, "rqp_yes", lang)
    # Acknowledge → closing
    result = await _step(machine, "qr_ack", lang)

    assert machine.is_complete
    assert machine.current_state == "closing"

    print("\n\n✅ Card not received path completed.")


@pytest.mark.anyio
async def test_bob_card_skip_benefits_activate_later() -> None:
    """Test skip benefits → activate later → SMS → closing."""
    lang = "en"
    machine = await _build_machine(lang)

    print("\n\n========== BOB CARD E2E — SKIP BENEFITS + LATER ==========")
    _print_node_status(machine, lang=lang)

    # Get through security + card check
    await _step(machine, "op_yes", lang)
    await _step(machine, "gtc_yes", lang)
    await _step(machine, "disc_proceed", lang)
    await _step(machine, "yob_given", lang)
    await _step(machine, "yob_echo_match", lang)
    await _step(machine, "l4_given", lang)
    await _step(machine, "l4_echo_match", lang)
    await _step(machine, "crc_yes", lang)

    # Skip benefits
    await _step(machine, "bo_skip", lang)

    # Activate later
    await _step(machine, "ap_later", lang)

    # Accept SMS link
    await _step(machine, "osl_yes", lang)

    # Acknowledge SMS sent
    await _step(machine, "ss_ack", lang)

    # No more queries → closing
    result = await _step(machine, "aqc_no", lang)

    assert machine.is_complete
    assert machine.current_state == "closing"

    print("\n\n✅ Skip benefits + activate later path completed.")


@pytest.mark.anyio
async def test_bob_card_language_filtering_at_each_node() -> None:
    """Verify language filtering works correctly at key bilingual nodes."""
    machine = await _build_machine("en")

    # Check a few key nodes with dual-language markers
    bilingual_nodes = [
        "good_time_check",
        "disclaimer",
        "yob_ask",
        "last4_ask",
        "card_received_check",
        "benefits_offer",
        "activation_prompt",
    ]

    print("\n\n========== LANGUAGE FILTERING CHECK ==========")
    for node_id in bilingual_nodes:
        node = machine._node_map[node_id]
        text = node.instruction or node.static_text or ""

        en_filtered = _filter_language_markers(text, "en")
        hi_filtered = _filter_language_markers(text, "hi")

        # Neither should contain raw [EN]/[HI] markers
        assert "[EN]" not in en_filtered, f"[EN] marker in EN text for {node_id}"
        assert "[HI]" not in en_filtered, f"[HI] marker in EN text for {node_id}"
        assert "[EN]" not in hi_filtered, f"[EN] marker in HI text for {node_id}"
        assert "[HI]" not in hi_filtered, f"[HI] marker in HI text for {node_id}"

        # EN and HI should produce different text
        assert en_filtered != hi_filtered, f"EN and HI text identical for {node_id}"

        print(
            f"\n{node_id}:\n  EN: {en_filtered[:100]}...\n  HI: {hi_filtered[:100]}..."
        )

    print("\n\n✅ Language filtering verified for all bilingual nodes.")


@pytest.mark.anyio
async def test_bob_card_tools_available_at_each_node() -> None:
    """Verify each node in the happy path has the right edge tools."""
    machine = await _build_machine("en")

    # Map of node → expected tool IDs
    expected_tools: dict[str, list[str]] = {
        "opening": ["op_yes", "op_wrong", "op_not_avail", "op_silent"],
        "good_time_check": ["gtc_yes", "gtc_busy", "gtc_silent"],
        "disclaimer": ["disc_proceed"],
        "yob_ask": ["yob_given", "yob_refused"],
        "yob_echo": [
            "yob_echo_match",
            "yob_echo_mismatch",
            "yob_echo_retry",
            "yob_echo_refused",
        ],
        "last4_ask": ["l4_given", "l4_refused"],
        "last4_echo": [
            "l4_echo_match",
            "l4_echo_mismatch",
            "l4_echo_retry",
            "l4_echo_refused",
        ],
        "card_received_check": ["crc_yes", "crc_no", "crc_wrong_addr"],
        "benefits_offer": ["bo_skip", "bo_yes", "bo_silent"],
        "benefits_details": ["bd_kb_question", "bd_proceed"],
        "activation_prompt": ["ap_yes", "ap_later", "ap_no"],
        "app_check": ["ac_yes", "ac_no"],
        "app_activation_guide": ["aag_done", "aag_later"],
        "activation_confirmed": ["aconf_ack"],
        "another_query_check": ["aqc_kb", "aqc_activation", "aqc_address", "aqc_no"],
    }

    print("\n\n========== TOOL AVAILABILITY CHECK ==========")
    for node_id, expected in expected_tools.items():
        node = machine._node_map[node_id]
        tools = machine.get_tools_for_node(node)
        tool_ids = [t.id for t in tools]

        assert set(expected) == set(
            tool_ids
        ), f"Tool mismatch for {node_id}: expected {expected}, got {tool_ids}"
        print(f"  {node_id}: {tool_ids} ✓")

    # Final nodes should have no tools
    final_nodes = [
        "closing",
        "wrong_person_end",
        "yob_mismatch_end",
        "yob_fail_end",
    ]
    for node_id in final_nodes:
        node = machine._node_map[node_id]
        tools = machine.get_tools_for_node(node)
        assert len(tools) == 0, f"Final node {node_id} should have no tools"
        print(f"  {node_id}: [] (final) ✓")

    print("\n\n✅ All tool availability checks passed.")
