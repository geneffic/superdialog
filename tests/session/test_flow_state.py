from superdialog.flow_state import FlowState
from superdialog.machine.models import FlowContext


def _populate(ctx: FlowContext) -> FlowContext:
    ctx.current_node_id = "collect_name"
    ctx.userdata = {"name": "Alice", "age": 30}
    ctx.node_slots = {"collect_name": {"name": "Alice"}}
    ctx.node_spoken_flags = {"collect_name": True, "greeting": True}
    ctx.visit_count = {"greeting": 1, "collect_name": 1}
    return ctx


def test_from_flow_context_captures_fields() -> None:
    ctx = _populate(FlowContext())
    state = FlowState.from_flow_context(ctx)
    assert state.current_node_id == "collect_name"
    assert state.userdata == {"name": "Alice", "age": 30}
    assert state.node_slots == {"collect_name": {"name": "Alice"}}
    assert state.node_spoken_flags == {"collect_name": True, "greeting": True}
    assert state.visit_count == {"greeting": 1, "collect_name": 1}


def test_apply_to_fresh_context_restores_fields() -> None:
    state = FlowState.from_flow_context(_populate(FlowContext()))
    fresh = FlowContext()
    state.apply_to(fresh)
    assert fresh.current_node_id == "collect_name"
    assert fresh.userdata == {"name": "Alice", "age": 30}
    assert fresh.node_slots == {"collect_name": {"name": "Alice"}}


def test_state_is_decoupled_from_source_context() -> None:
    src = _populate(FlowContext())
    state = FlowState.from_flow_context(src)
    src.userdata["mutated"] = True
    assert "mutated" not in state.userdata


def test_to_flow_context_merged_with_alias() -> None:
    state = FlowState.from_flow_context(_populate(FlowContext()))
    fresh = FlowContext()
    merged = state.to_flow_context_merged_with(fresh)
    assert merged is fresh
    assert merged.current_node_id == "collect_name"
