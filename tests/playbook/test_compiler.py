"""Tests for the flow→playbook compiler (FlowIndex node classification)."""

import json
from pathlib import Path

from superdialog.flow.models import ConversationFlow, Edge, FlowNode
from superdialog.playbook.compiler import (
    FlowIndex,
    compile_edge_condition,
    union_slot_schemas,
)

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


def test_data_predicate_conditions_become_expr_rules() -> None:
    rule = compile_edge_condition(
        "availability_result.success == true",
        store_keys={"availability_result"},
        target="main.present",
    )
    assert rule.judge == "expr"
    assert rule.when == "results.availability_result.ok"
    assert rule.to == "main.present"


def test_status_code_predicates_translate() -> None:
    rule = compile_edge_condition(
        "hold_result.status == 409", store_keys={"hold_result"}, target="main.taken"
    )
    assert rule.judge == "expr"
    assert rule.when == "results.hold_result.status == 409"


def test_intent_conditions_become_llm_rules() -> None:
    rule = compile_edge_condition(
        "caller wants to cancel an existing booking",
        store_keys=set(),
        target="main.cancel",
    )
    assert rule.judge == "llm"
    assert rule.when == "caller wants to cancel an existing booking"


def test_untranslatable_data_conditions_stay_llm() -> None:
    # mentions a store key but the shape isn't confidently translatable
    rule = compile_edge_condition(
        "availability_result has slots near the preferred time",
        store_keys={"availability_result"},
        target="main.present",
    )
    assert rule.judge == "llm"


def test_negated_and_prose_success_forms() -> None:
    keys = {"hold_result", "booking_confirm_result"}
    negated = compile_edge_condition(
        "hold_result.success == false", store_keys=keys, target="main.t"
    )
    assert (negated.judge, negated.when) == ("expr", "not results.hold_result.ok")
    not_form = compile_edge_condition(
        "not hold_result.success", store_keys=keys, target="main.t"
    )
    assert (not_form.judge, not_form.when) == ("expr", "not results.hold_result.ok")
    # legacy prose spelling used by the golf flow, with an em-dash gloss
    prose = compile_edge_condition(
        "booking_confirm_result.success is false — route to retry attempt",
        store_keys=keys,
        target="main.retry",
    )
    assert prose.judge == "expr"
    assert prose.when == "not results.booking_confirm_result.ok"
    # a gloss that qualifies the predicate must NOT be dropped
    qualified = compile_edge_condition(
        "hold_result.success is true — unless the caller already paid",
        store_keys=keys,
        target="main.t",
    )
    assert qualified.judge == "llm"


def test_unknown_store_key_and_compounds_stay_llm() -> None:
    unknown = compile_edge_condition(
        "mystery_result.success == true", store_keys={"hold_result"}, target="main.t"
    )
    assert unknown.judge == "llm"
    assert unknown.when == "mystery_result.success == true"
    compound = compile_edge_condition(
        "hold_result.success == true and availability_result.success == true",
        store_keys={"hold_result", "availability_result"},
        target="main.t",
    )
    assert compound.judge == "llm"


def test_union_schemas_with_per_rule_requires() -> None:
    flow = _flow()
    node = next(n for n in flow.nodes if n.id == "collect_booking_details")
    slots, requires_by_edge = union_slot_schemas(node)
    edges_with_schema = {e.id for e in node.edges if e.input_schema}
    assert set(requires_by_edge) == edges_with_schema
    for req in requires_by_edge.values():
        assert set(req) <= set(slots)  # every required is declared
    assert all(not s.required for s in slots.values())  # union: all optional


def test_json_schema_type_mapping() -> None:
    flow = _flow()
    # find an array-typed property anywhere in the flow
    found_array = False
    for node in flow.nodes:
        for e in node.edges:
            if not e.input_schema:
                continue
            slots, _ = union_slot_schemas(node)
            assert isinstance(e.input_schema, dict)
            for key, prop in (e.input_schema.get("properties") or {}).items():
                if prop.get("type") == "array":
                    assert slots[key].type == "array"
                    found_array = True
    assert found_array  # the golf flow has 2 array fields


def test_enum_description_and_first_declaration_wins() -> None:
    node = FlowNode(
        id="n",
        name="n",
        edges=[
            Edge(
                id="e1",
                condition="c1",
                target_node_id="x",
                input_schema={
                    "type": "object",
                    "properties": {
                        "tee_period": {
                            "type": "string",
                            "enum": ["morning", "afternoon"],
                            "description": "Preferred period",
                        },
                        "count": {"type": "integer"},
                    },
                    "required": ["tee_period"],
                },
            ),
            Edge(
                id="e2",
                condition="c2",
                target_node_id="y",
                input_schema={
                    "type": "object",
                    "properties": {"tee_period": {"type": "string"}},
                    "required": [],
                },
            ),
        ],
    )
    slots, requires_by_edge = union_slot_schemas(node)
    assert slots["tee_period"].type == "enum"  # first declaration wins
    assert slots["tee_period"].values == ["morning", "afternoon"]
    assert slots["tee_period"].description == "Preferred period"
    assert slots["count"].type == "int"
    assert requires_by_edge == {"e1": ["tee_period"], "e2": []}
