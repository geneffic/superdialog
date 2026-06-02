"""Spec-aligned helpers on top of ConversationFlow."""

from __future__ import annotations

from pathlib import Path

from .models import ConversationFlow as Flow


class FlowSet:
    """A named collection of Flows."""

    def __init__(self, flows: dict[str, Flow]):
        self.flows = flows

    def __getitem__(self, name: str) -> Flow:
        return self.flows[name]

    def __contains__(self, name: object) -> bool:
        return name in self.flows

    def names(self) -> list[str]:
        return list(self.flows.keys())


def save_flow(flow: Flow, path: str | Path) -> None:
    """Serialize ``flow`` to ``path`` as indented JSON.

    Backward-compatible wrapper around :meth:`ConversationFlow.save`.
    """
    flow.save(path)


def load_flow(path: str | Path) -> Flow:
    """Load a Flow from a JSON file at ``path``.

    Backward-compatible wrapper around :meth:`ConversationFlow.load`.
    """
    return Flow.load(path)
