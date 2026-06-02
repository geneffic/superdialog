"""DialogStateMachine engine — ported from super.core.voice.dialog_machine.

This module is the internal engine; the recommended public facade is
``superdialog.DialogMachine`` (added in Task 5).
"""

from .actions import ActionExecutor
from .criteria import CriteriaJudge
from .extractor import VariableExtractor
from .gate import TransitionGate
from .hooks import MachineHooks
from .machine import DialogStateMachine
from .models import (
    CriteriaResult,
    FlowContext,
    NodeScope,
    ToolDescriptor,
    TransitionRecord,
    TurnResult,
)
from .runner import create_machine, run_flow
from .store import ContextStore, InMemoryContextStore

__all__ = [
    "ActionExecutor",
    "ContextStore",
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
    "create_machine",
    "run_flow",
]
