"""Tests for the flow→playbook compiler (FlowIndex node classification)."""

import json
from pathlib import Path

from superdialog.flow.models import ConversationFlow
from superdialog.playbook.compiler import FlowIndex

GOLF = Path(__file__).parents[1] / "fixtures" / "flow" / "golf_booking.json"

EXPECTED_SYSTEM = {
    "check_booking_status",
    "payment_expired_handler",
    "webhook_booking_confirm",
    "not_registered_close",
}
EXPECTED_COMPUTATIONAL = {
    "player_id_check",
    "greeting_details",
    "token_refresh",
    "resolve_course_name",
    "check_course_availability",
    "profile_check",
    "create_player_profile",
    "hold_slot_payment",
    "confirm_booking",
    "confirm_booking_retry",
    "slot_taken",
    "profile_check_waitlist",
    "create_player_for_waitlist",
}


def _flow() -> ConversationFlow:
    return ConversationFlow.model_validate(json.loads(GOLF.read_text()))


def test_classification_matches_derived_ground_truth() -> None:
    idx = FlowIndex(_flow())
    kinds = {n.id: idx.classify(n) for n in idx.flow.nodes}
    assert {n for n, k in kinds.items() if k == "system"} == EXPECTED_SYSTEM
    assert {
        n for n, k in kinds.items() if k == "computational"
    } == EXPECTED_COMPUTATIONAL
    assert kinds["collect_booking_details"] == "conversational"
    assert kinds["silence_check"] == "conversational"


def test_reverse_edges_and_node_lookup() -> None:
    idx = FlowIndex(_flow())
    inbound = idx.reverse_edges["collect_booking_details"]
    assert len(inbound) == 15
    assert all(isinstance(src, str) and isinstance(eid, str) for src, eid in inbound)
    node = idx.node("collect_booking_details")
    assert node.id == "collect_booking_details"
    assert idx.node("token_refresh").id == "token_refresh"
