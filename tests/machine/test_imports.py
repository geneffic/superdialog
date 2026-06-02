"""Verify the ported engine modules import cleanly."""

import superdialog.machine as machine_pkg
from superdialog.machine import DialogStateMachine, FlowContext, InMemoryContextStore


def test_engine_imports():
    """All public names declared in __all__ resolve from the package."""
    for name in (
        "ActionExecutor",
        "CriteriaJudge",
        "CriteriaResult",
        "DialogStateMachine",
        "FlowContext",
        "InMemoryContextStore",
        "MachineHooks",
        "NodeScope",
        "ToolDescriptor",
        "TransitionGate",
        "TransitionRecord",
        "TurnResult",
        "VariableExtractor",
    ):
        assert getattr(machine_pkg, name) is not None, name

    assert DialogStateMachine is not None
    store = InMemoryContextStore()
    assert store is not None


def test_context_construction():
    ctx = FlowContext(current_node_id="root")
    assert ctx.current_node_id == "root"


def test_dialog_state_machine_class_is_importable():
    """Smoke: DialogStateMachine class can be referenced without instantiation
    (full construction requires an adapter and a flow; that's covered in
    Task 7)."""
    from superdialog.machine import DialogStateMachine

    assert hasattr(DialogStateMachine, "from_flow")
    assert callable(DialogStateMachine.from_flow)
