"""Tests for dialog_machine.models."""

import time

from superdialog.machine.models import CriteriaResult, FlowContext, TransitionRecord


class TestCriteriaResult:
    """Tests for CriteriaResult model."""

    def test_defaults(self) -> None:
        result = CriteriaResult(node_id="n1")
        assert result.node_id == "n1"
        assert result.criteria_met == {}
        assert result.all_required_met is False
        assert result.user_insisting is False
        assert result.recommended_edge_id is None
        assert result.reason == ""

    def test_full_construction(self) -> None:
        result = CriteriaResult(
            node_id="n2",
            criteria_met={"q1": True, "q2": False},
            all_required_met=False,
            user_insisting=True,
            recommended_edge_id="e1",
            reason="user wants to skip",
        )
        assert result.criteria_met == {"q1": True, "q2": False}
        assert result.user_insisting is True
        assert result.recommended_edge_id == "e1"

    def test_serialization_roundtrip(self) -> None:
        result = CriteriaResult(
            node_id="n1",
            criteria_met={"a": True},
            all_required_met=True,
            recommended_edge_id="e1",
            reason="done",
        )
        data = result.model_dump()
        restored = CriteriaResult.model_validate(data)
        assert restored == result


class TestTransitionRecord:
    """Tests for TransitionRecord model."""

    def test_defaults(self) -> None:
        before = time.time()
        record = TransitionRecord(from_node="a", to_node="b", edge_id="e1")
        after = time.time()
        assert record.from_node == "a"
        assert record.to_node == "b"
        assert record.edge_id == "e1"
        assert record.criteria_met == {}
        assert record.skipped is False
        assert before <= record.timestamp <= after

    def test_with_criteria(self) -> None:
        record = TransitionRecord(
            from_node="a",
            to_node="b",
            edge_id="e1",
            criteria_met={"q1": True},
            skipped=True,
        )
        assert record.criteria_met == {"q1": True}
        assert record.skipped is True

    def test_serialization_roundtrip(self) -> None:
        record = TransitionRecord(
            from_node="a",
            to_node="b",
            edge_id="e1",
            criteria_met={"x": False},
            skipped=True,
            timestamp=1000.0,
        )
        data = record.model_dump()
        restored = TransitionRecord.model_validate(data)
        assert restored == record


def test_transition_record_carries_messages():
    from superdialog.machine.models import TransitionRecord

    rec = TransitionRecord(
        from_node="a",
        to_node="b",
        edge_id="a_to_b",
        user_message="hello there",
        bot_message="Hi! What can I do?",
    )
    assert rec.user_message == "hello there"
    assert rec.bot_message == "Hi! What can I do?"

    # Defaults keep old call sites valid (back-compat for persisted logs).
    bare = TransitionRecord(from_node="a", to_node="b", edge_id="a_to_b")
    assert bare.user_message is None
    assert bare.bot_message == ""


class TestFlowContext:
    """Tests for FlowContext model."""

    def test_defaults(self) -> None:
        ctx = FlowContext()
        assert ctx.conversation_history == []
        assert ctx.userdata == {}
        assert ctx.criteria_status == {}
        assert ctx.transition_log == []
        assert ctx.current_node_id == ""

    def test_add_message(self) -> None:
        ctx = FlowContext()
        ctx.add_message("user", "hello")
        ctx.add_message("assistant", "hi there")
        assert len(ctx.conversation_history) == 2
        assert ctx.conversation_history[0] == {
            "role": "user",
            "content": "hello",
        }
        assert ctx.conversation_history[1] == {
            "role": "assistant",
            "content": "hi there",
        }

    def test_serialization_roundtrip(self) -> None:
        ctx = FlowContext(
            current_node_id="n1",
            userdata={"key": "val"},
        )
        ctx.add_message("user", "test")
        ctx.transition_log.append(
            TransitionRecord(
                from_node="a",
                to_node="b",
                edge_id="e1",
                timestamp=1000.0,
            )
        )
        data = ctx.model_dump()
        restored = FlowContext.model_validate(data)
        assert restored.current_node_id == "n1"
        assert len(restored.conversation_history) == 1
        assert len(restored.transition_log) == 1
        assert restored.userdata == {"key": "val"}

    def test_independent_instances(self) -> None:
        """Ensure default factories create independent lists/dicts."""
        ctx1 = FlowContext()
        ctx2 = FlowContext()
        ctx1.add_message("user", "only in ctx1")
        ctx1.userdata["x"] = 1
        assert ctx2.conversation_history == []
        assert ctx2.userdata == {}
