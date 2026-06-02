"""End-to-end Kairali flow traversal tests using DialogStateMachine.

Ports ALL kairali paths from test_flow_agent_transitions.py to use
DialogStateMachine with MockAdapter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

for _mod in [
    "livekit.agents",
    "livekit.agents.llm",
    "livekit.agents.llm.tool_context",
    "livekit.agents.voice",
    "livekit.api",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from superdialog.flow.models import ConversationFlow  # noqa: E402
from superdialog.machine.machine import DialogStateMachine  # noqa: E402
from superdialog.machine.testing.mock_adapter import MockAdapter  # noqa: E402

TEMP_DIR = Path(__file__).resolve().parents[4] / "temp"


@pytest.fixture
def kairali_flow() -> ConversationFlow:
    path = TEMP_DIR / "kairali_flow 2.json"
    if not path.exists():
        pytest.skip("kairali_flow 2.json not found")
    return ConversationFlow.from_json_file(path)


async def _drive(flow: ConversationFlow, edges: list[str]) -> DialogStateMachine:
    """Drive a DialogStateMachine through the given edge sequence."""
    adapter = MockAdapter(edges)
    machine = await DialogStateMachine.from_flow(flow, adapter)
    for _ in edges:
        await machine.process_turn("input")
    return machine


def _visited(machine: DialogStateMachine) -> list[str]:
    """Extract ordered list of visited node IDs from transition log."""
    log = machine.context.transition_log
    if not log:
        return [machine._flow.initial_node]
    nodes = [log[0].from_node]
    for record in log:
        nodes.append(record.to_node)
    return nodes


# =========================================================================
# Quick Exits
# =========================================================================


class TestKairaliQuickExits:
    """Short paths that exit early without reaching enquiry_open."""

    @pytest.mark.anyio
    async def test_wrong_number(self, kairali_flow: ConversationFlow) -> None:
        """greeting -> unqualified_close (F)"""
        machine = await _drive(kairali_flow, ["greeting_wrong_number"])
        visited = _visited(machine)
        assert visited == ["greeting", "unqualified_close"]
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_already_spoke(self, kairali_flow: ConversationFlow) -> None:
        """greeting -> already_spoke_close (F)"""
        machine = await _drive(kairali_flow, ["greeting_already_spoke"])
        visited = _visited(machine)
        assert visited == ["greeting", "already_spoke_close"]
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_person_not_available(self, kairali_flow: ConversationFlow) -> None:
        """greeting -> person_not_available -> person_not_available_confirm (F)"""
        machine = await _drive(
            kairali_flow,
            ["greeting_not_available", "person_na_time_given"],
        )
        visited = _visited(machine)
        assert visited == [
            "greeting",
            "person_not_available",
            "person_not_available_confirm",
        ]
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_did_not_enquire_nobody(self, kairali_flow: ConversationFlow) -> None:
        """greeting -> did_not_enquire_check -> nobody_enquired_close (F)"""
        machine = await _drive(
            kairali_flow,
            ["greeting_did_not_enquire", "nobody_enquired"],
        )
        visited = _visited(machine)
        assert visited[-1] == "nobody_enquired_close"
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_not_interested(self, kairali_flow: ConversationFlow) -> None:
        """greeting -> availability -> not_interested -> not_interested_close (F)"""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_not_interested",
                "not_interested_reason_given",
            ],
        )
        visited = _visited(machine)
        assert visited == [
            "greeting",
            "availability_check",
            "not_interested_reason",
            "not_interested_close",
        ]
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_reschedule(self, kairali_flow: ConversationFlow) -> None:
        """greeting -> availability -> reschedule -> reschedule_confirm (F)"""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_no_reschedule",
                "reschedule_time_given",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "reschedule_confirm"
        assert machine.is_complete


# =========================================================================
# Product Paths
# =========================================================================


class TestKairaliProductPaths:
    """Products self-use and business paths through contact capture."""

    @pytest.mark.anyio
    async def test_products_self_use_full_path(
        self, kairali_flow: ConversationFlow
    ) -> None:
        """Full products self-use path through contact capture to closing."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_products",
                "products_self_use",
                "products_self_use_captured",
                "products_self_to_contact",
                "alt_number_given",
                "email_given",
                "city_given",
                "callback_time_given",
                "contact_confirm_to_close",
            ],
        )
        visited = _visited(machine)
        assert visited == [
            "greeting",
            "availability_check",
            "enquiry_open",
            "products_self_or_business",
            "products_self_use_capture",
            "products_self_use_callback",
            "contact_capture",
            "contact_capture_email",
            "contact_capture_city",
            "contact_capture_time",
            "contact_capture_confirm",
            "closing",
        ]
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_products_business_path(self, kairali_flow: ConversationFlow) -> None:
        """Products business path -> contact capture -> closing."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_products",
                "products_business",
                "products_business_captured",
                "products_business_to_contact",
                "alt_number_skipped",
                "email_skipped",
                "city_skipped",
                "callback_time_skipped",
                "contact_confirm_to_close",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "closing"
        assert machine.is_complete


# =========================================================================
# Service Paths
# =========================================================================


class TestKairaliServicePaths:
    """Service-related paths: healing village, villa raag, etc."""

    @pytest.mark.anyio
    async def test_healing_village_treatment(
        self, kairali_flow: ConversationFlow
    ) -> None:
        """Healing Village treatment path through booking to closing."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_healing_village",
                "hv_treatment",
                "hv_condition_captured",
                "hv_booking_type_captured",
                "hv_to_contact",
                "alt_number_given",
                "email_given",
                "city_given",
                "callback_time_given",
                "contact_confirm_to_close",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "closing"
        assert "healing_village_type" in visited
        assert "healing_village_treatment_capture" in visited
        assert "healing_village_booking_type" in visited
        assert "healing_village_callback" in visited
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_healing_village_preventive(
        self, kairali_flow: ConversationFlow
    ) -> None:
        """Healing Village preventive skips treatment capture."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_healing_village",
                "hv_preventive",
                "hv_booking_type_captured",
                "hv_to_contact",
            ],
        )
        visited = _visited(machine)
        assert "healing_village_treatment_capture" not in visited
        assert "healing_village_booking_type" in visited

    @pytest.mark.anyio
    async def test_villa_raag(self, kairali_flow: ConversationFlow) -> None:
        """Villa Raag guest type -> purpose -> callback -> contact -> closing."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_villa_raag",
                "villa_raag_guest_captured",
                "villa_raag_purpose_captured",
                "villa_to_contact",
                "alt_number_given",
                "email_given",
                "city_given",
                "callback_time_given",
                "contact_confirm_to_close",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "closing"
        assert "villa_raag_guest_type" in visited
        assert "villa_raag_purpose" in visited
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_treatment_centre(self, kairali_flow: ConversationFlow) -> None:
        """Treatment centre -> callback -> contact capture -> closing."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_treatment_centre",
                "treatment_centre_captured",
                "treatment_to_contact",
                "alt_number_skipped",
                "email_skipped",
                "city_given",
                "callback_time_given",
                "contact_confirm_to_close",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "closing"
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_training(self, kairali_flow: ConversationFlow) -> None:
        """Training -> callback -> contact capture -> closing."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_training",
                "training_type_captured",
                "training_to_contact",
                "alt_number_given",
                "email_given",
                "city_given",
                "callback_time_given",
                "contact_confirm_to_close",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "closing"
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_franchise(self, kairali_flow: ConversationFlow) -> None:
        """Franchise -> callback -> contact capture -> closing."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_franchise",
                "franchise_type_captured",
                "franchise_to_contact",
                "alt_number_given",
                "email_given",
                "city_given",
                "callback_time_given",
                "contact_confirm_to_close",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "closing"
        assert machine.is_complete


# =========================================================================
# Order Paths
# =========================================================================


class TestKairaliOrderPaths:
    """Order-related paths: dispatched, no realtime, no ID."""

    @pytest.mark.anyio
    async def test_order_dispatched(self, kairali_flow: ConversationFlow) -> None:
        """Order with ID -> dispatched -> closing (no contact capture)."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_order",
                "order_id_provided",
                "order_is_dispatched",
                "order_dispatched_close",
            ],
        )
        visited = _visited(machine)
        assert visited == [
            "greeting",
            "availability_check",
            "enquiry_open",
            "order_id_check",
            "order_status_check",
            "order_dispatched",
            "closing",
        ]
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_order_no_realtime(self, kairali_flow: ConversationFlow) -> None:
        """Order with no real-time status -> contact capture -> closing."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_order",
                "order_id_provided",
                "order_status_unclear",
                "order_no_realtime_to_contact",
                "alt_number_given",
                "email_given",
                "city_given",
                "callback_time_given",
                "contact_confirm_to_close",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "closing"
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_order_no_id(self, kairali_flow: ConversationFlow) -> None:
        """Order without ID -> no realtime -> contact capture."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_order",
                "order_id_not_available",
                "order_no_realtime_to_contact",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "contact_capture"


# =========================================================================
# Misc Paths
# =========================================================================


class TestKairaliMiscPaths:
    """Careers, general info, other query, structural checks."""

    @pytest.mark.anyio
    async def test_careers_with_callback(self, kairali_flow: ConversationFlow) -> None:
        """Careers with callback -> contact capture -> closing."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_careers",
                "careers_wants_callback",
                "alt_number_given",
                "email_given",
                "city_given",
                "callback_time_given",
                "contact_confirm_to_close",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "closing"
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_careers_no_callback(self, kairali_flow: ConversationFlow) -> None:
        """Careers without callback -> closing directly."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_careers",
                "careers_no_callback",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "closing"
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_general_info_loop_back(self, kairali_flow: ConversationFlow) -> None:
        """General info -> wants more -> back to enquiry_open (loop)."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_general_kairali",
                "general_info_wants_more",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "enquiry_open"
        # enquiry_open appears twice: once from availability, once from loop
        assert visited.count("enquiry_open") == 2

    @pytest.mark.anyio
    async def test_general_info_satisfied(self, kairali_flow: ConversationFlow) -> None:
        """General info -> satisfied -> closing."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_general_kairali",
                "general_info_satisfied",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "closing"
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_other_query(self, kairali_flow: ConversationFlow) -> None:
        """Other query -> contact capture -> closing."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_yes",
                "available_yes",
                "enquiry_other",
                "other_query_captured",
                "alt_number_skipped",
                "email_skipped",
                "city_skipped",
                "callback_time_skipped",
                "contact_confirm_to_close",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "closing"
        assert machine.is_complete

    @pytest.mark.anyio
    async def test_family_enquiry_to_open(self, kairali_flow: ConversationFlow) -> None:
        """did_not_enquire -> family -> enquiry_open."""
        machine = await _drive(
            kairali_flow,
            [
                "greeting_did_not_enquire",
                "family_did_enquire",
                "family_continue_to_open",
            ],
        )
        visited = _visited(machine)
        assert visited[-1] == "enquiry_open"

    def test_all_final_nodes(self, kairali_flow: ConversationFlow) -> None:
        """Verify all expected final nodes are marked is_final."""
        expected_finals = {
            "person_not_available_confirm",
            "already_spoke_close",
            "nobody_enquired_close",
            "reschedule_confirm",
            "not_interested_close",
            "unqualified_close",
            "closing",
        }
        actual_finals = {n.id for n in kairali_flow.nodes if n.is_final}
        assert expected_finals == actual_finals

    def test_enquiry_open_has_10_edges(self, kairali_flow: ConversationFlow) -> None:
        """enquiry_open has 10 edges -- it is the main routing hub."""
        node = next(n for n in kairali_flow.nodes if n.id == "enquiry_open")
        assert len(node.edges) == 10
        edge_ids = {e.id for e in node.edges}
        expected = {
            "enquiry_products",
            "enquiry_healing_village",
            "enquiry_villa_raag",
            "enquiry_treatment_centre",
            "enquiry_training",
            "enquiry_order",
            "enquiry_franchise",
            "enquiry_careers",
            "enquiry_general_kairali",
            "enquiry_other",
        }
        assert edge_ids == expected
