"""Tests for MockAdapter and MockAdapterWithCriteria."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from superdialog.machine.testing.mock_adapter import (
    MockAdapter,
    MockAdapterWithCriteria,
)


@dataclass
class _FakeNode:
    """Lightweight stand-in for FlowNode (avoids livekit import chain)."""

    id: str = "n1"
    name: str = "Node n1"
    instruction: str | None = None
    static_text: str | None = None
    is_final: bool = False
    edges: list[object] = field(default_factory=list)
    actions: list[object] = field(default_factory=list)


@dataclass
class _FakeAction:
    """Lightweight stand-in for CustomAction."""

    id: str = "a1"
    name: str = "Action a1"
    description: str = "test action"
    method: str = "GET"
    url: str = "https://example.com"


def _make_node(node_id: str = "n1") -> _FakeNode:
    return _FakeNode(id=node_id, name=f"Node {node_id}")


def _make_action(action_id: str = "a1") -> _FakeAction:
    return _FakeAction(id=action_id, name=f"Action {action_id}")


class TestMockAdapter:
    """Tests for MockAdapter."""

    @pytest.mark.anyio
    async def test_speak_records_text(self) -> None:
        adapter = MockAdapter(edge_sequence=[])
        node = _make_node()
        await adapter.speak("hello", node)  # type: ignore[arg-type]
        await adapter.speak("world", node)  # type: ignore[arg-type]
        assert adapter.spoken == ["hello", "world"]

    @pytest.mark.anyio
    async def test_generate_reply(self) -> None:
        adapter = MockAdapter(edge_sequence=[])
        node = _make_node("greeting")
        reply = await adapter.generate_reply("say hi", node)  # type: ignore[arg-type]
        assert "greeting" in reply
        assert adapter.replies == [reply]

    @pytest.mark.anyio
    async def test_evaluate_criteria_sequence(self) -> None:
        adapter = MockAdapter(edge_sequence=["e1", "e2", "e3"])
        node = _make_node()

        r1 = await adapter.evaluate_criteria(node, [], {})  # type: ignore[arg-type]
        assert r1.all_required_met is True
        assert r1.recommended_edge_id == "e1"

        r2 = await adapter.evaluate_criteria(node, [], {})  # type: ignore[arg-type]
        assert r2.recommended_edge_id == "e2"

        r3 = await adapter.evaluate_criteria(node, [], {})  # type: ignore[arg-type]
        assert r3.recommended_edge_id == "e3"

        # Exhausted
        r4 = await adapter.evaluate_criteria(node, [], {})  # type: ignore[arg-type]
        assert r4.all_required_met is False
        assert r4.recommended_edge_id is None

    @pytest.mark.anyio
    async def test_execute_action_records(self) -> None:
        adapter = MockAdapter(edge_sequence=[])
        action = _make_action("act1")
        result = await adapter.execute_action(action, {})  # type: ignore[arg-type]
        assert result is None
        assert adapter.actions_executed == ["act1"]

    @pytest.mark.anyio
    async def test_end_session(self) -> None:
        adapter = MockAdapter(edge_sequence=[])
        assert adapter.session_ended is False
        await adapter.end_session()
        assert adapter.session_ended is True


class TestMockAdapterWithCriteria:
    """Tests for MockAdapterWithCriteria."""

    @pytest.mark.anyio
    async def test_fixed_criteria(self) -> None:
        adapter = MockAdapterWithCriteria(
            edge_id="e_fixed",
            criteria_met={"q1": True, "q2": True},
        )
        node = _make_node()
        result = await adapter.evaluate_criteria(node, [], {})  # type: ignore[arg-type]
        assert result.all_required_met is True
        assert result.recommended_edge_id == "e_fixed"
        assert result.criteria_met == {"q1": True, "q2": True}

    @pytest.mark.anyio
    async def test_partial_criteria(self) -> None:
        adapter = MockAdapterWithCriteria(
            edge_id="e_partial",
            criteria_met={"q1": True, "q2": False},
        )
        node = _make_node()
        result = await adapter.evaluate_criteria(node, [], {})  # type: ignore[arg-type]
        assert result.all_required_met is False

    @pytest.mark.anyio
    async def test_user_insisting(self) -> None:
        adapter = MockAdapterWithCriteria(
            edge_id="e1",
            user_insisting=True,
        )
        node = _make_node()
        result = await adapter.evaluate_criteria(node, [], {})  # type: ignore[arg-type]
        assert result.user_insisting is True

    @pytest.mark.anyio
    async def test_speak_and_end_session(self) -> None:
        adapter = MockAdapterWithCriteria(edge_id="e1")
        node = _make_node()
        await adapter.speak("test", node)  # type: ignore[arg-type]
        assert adapter.spoken == ["test"]
        await adapter.end_session()
        assert adapter.session_ended is True

    @pytest.mark.anyio
    async def test_consistent_results(self) -> None:
        """Calling evaluate_criteria multiple times returns same result."""
        adapter = MockAdapterWithCriteria(
            edge_id="e_stable",
            criteria_met={"done": True},
        )
        node = _make_node()
        r1 = await adapter.evaluate_criteria(node, [], {})  # type: ignore[arg-type]
        r2 = await adapter.evaluate_criteria(node, [], {})  # type: ignore[arg-type]
        assert r1.recommended_edge_id == r2.recommended_edge_id
        assert r1.all_required_met == r2.all_required_met
