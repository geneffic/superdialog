"""Tests for the conversation engine rewrite of DialogStateMachine.

Covers: TurnResult contract, stay/error paths, action execution,
re-entry tracking, context persistence, LLM retry, and history recording.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

for _mod in [
    "livekit.agents",
    "livekit.agents.llm",
    "livekit.agents.llm.tool_context",
    "livekit.agents.voice",
    "livekit.api",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402

from superdialog.flow.models import (  # noqa: E402
    ActionTrigger,
    ActionTriggerType,
    ConversationFlow,
    CustomAction,
    Edge,
    FlowNode,
    HttpMethod,
)
from superdialog.machine.machine import DialogStateMachine  # noqa: E402
from superdialog.machine.models import CriteriaResult, TurnResult  # noqa: E402
from superdialog.machine.store import InMemoryContextStore  # noqa: E402

# ---------------------------------------------------------------------------
# Test Adapter — full control over behavior
# ---------------------------------------------------------------------------


class ControlAdapter:
    """Adapter that returns whatever CriteriaResult you configure."""

    def __init__(self) -> None:
        self.results: list[CriteriaResult] = []
        self._index = 0
        self.spoken: list[str] = []
        self.replies: list[str] = []
        self.actions_executed: list[str] = []
        self.session_ended: bool = False
        self.recovery_calls: list[str] = []

    def queue_result(self, result: CriteriaResult) -> None:
        self.results.append(result)

    async def speak(self, text: str, node: FlowNode) -> None:
        self.spoken.append(text)

    async def generate_reply(self, instruction: str, node: FlowNode, history=None, userdata=None) -> str:
        reply = f"reply:{node.id}"
        self.replies.append(reply)
        return reply

    async def evaluate_criteria(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
    ) -> CriteriaResult:
        if self._index < len(self.results):
            r = self.results[self._index]
            self._index += 1
            return r
        return CriteriaResult(
            node_id=node.id,
            all_required_met=False,
            reason="no result queued",
            response="default stay response",
        )

    async def generate_recovery(self, node: FlowNode, error: str) -> str:
        self.recovery_calls.append(error)
        return f"recovery:{node.id}"

    async def execute_action(
        self,
        action: CustomAction,
        userdata: dict[str, Any],
    ) -> dict[str, Any] | None:
        self.actions_executed.append(action.id)
        if action.store_response_as:
            return {"status": 200, "data": f"result_{action.id}"}
        return None

    async def end_session(self) -> None:
        self.session_ended = True


class ErrorAdapter(ControlAdapter):
    """Adapter whose evaluate_criteria always raises."""

    async def evaluate_criteria(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
    ) -> CriteriaResult:
        raise RuntimeError("LLM exploded")


# ---------------------------------------------------------------------------
# Flow Fixtures
# ---------------------------------------------------------------------------


def _simple_flow() -> ConversationFlow:
    return ConversationFlow(
        system_prompt="test",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="Ask user something",
                edges=[
                    Edge(
                        id="e_start_mid",
                        condition="user ready",
                        target_node_id="middle",
                    ),
                ],
            ),
            FlowNode(
                id="middle",
                name="Middle",
                instruction="Collect details",
                edges=[
                    Edge(
                        id="e_mid_end",
                        condition="user done",
                        target_node_id="end",
                    ),
                ],
            ),
            FlowNode(
                id="end",
                name="End",
                static_text="Goodbye!",
                is_final=True,
            ),
        ],
    )


def _flow_with_actions() -> ConversationFlow:
    return ConversationFlow(
        system_prompt="test",
        initial_node="start",
        actions=[
            CustomAction(
                id="act_1",
                name="Action 1",
                description="Test action",
                method=HttpMethod.GET,
                url="http://example.com",
                store_response_as="api_result",
            ),
            CustomAction(
                id="act_2",
                name="Action 2",
                description="Test action 2",
                method=HttpMethod.POST,
                url="http://example.com",
            ),
        ],
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="Go",
                edges=[
                    Edge(
                        id="e_to_end",
                        condition="done",
                        target_node_id="end",
                        actions=[
                            ActionTrigger(
                                trigger_type=ActionTriggerType.ON_ENTER,
                                action_id="act_1",
                            ),
                        ],
                    ),
                ],
            ),
            FlowNode(
                id="end",
                name="End",
                static_text="Done!",
                is_final=True,
                actions=[
                    ActionTrigger(
                        trigger_type=ActionTriggerType.ON_ENTER,
                        action_id="act_2",
                    ),
                ],
            ),
        ],
    )


def _flow_with_fallback() -> ConversationFlow:
    return ConversationFlow(
        system_prompt="test",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="Ask",
                max_turns=2,
                edges=[
                    Edge(
                        id="e_normal",
                        condition="user answered",
                        target_node_id="end",
                    ),
                    Edge(
                        id="e_fallback",
                        condition="fallback",
                        target_node_id="end",
                        is_fallback=True,
                    ),
                ],
            ),
            FlowNode(
                id="end",
                name="End",
                static_text="Done!",
                is_final=True,
            ),
        ],
    )


def _loop_flow() -> ConversationFlow:
    """Flow with a loop: start → middle → start."""
    return ConversationFlow(
        system_prompt="test",
        initial_node="start",
        nodes=[
            FlowNode(
                id="start",
                name="Start",
                instruction="Ask",
                edges=[
                    Edge(
                        id="e_to_mid",
                        condition="go",
                        target_node_id="middle",
                    ),
                ],
            ),
            FlowNode(
                id="middle",
                name="Middle",
                instruction="Middle step",
                edges=[
                    Edge(
                        id="e_back",
                        condition="back",
                        target_node_id="start",
                    ),
                    Edge(
                        id="e_to_end",
                        condition="done",
                        target_node_id="end",
                    ),
                ],
            ),
            FlowNode(
                id="end",
                name="End",
                static_text="Bye!",
                is_final=True,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 7.1 - TurnResult on transition
# ---------------------------------------------------------------------------


class TestTransitionTurnResult:
    @pytest.mark.anyio
    async def test_transition_returns_turn_result(self) -> None:
        adapter = ControlAdapter()
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                criteria_met={"ready": True},
                all_required_met=True,
                recommended_edge_id="e_start_mid",
                reason="ready",
            )
        )
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)

        result = await machine.process_turn("go")

        assert isinstance(result, TurnResult)
        assert result.outcome == "transition"
        assert result.from_node == "start"
        assert result.to_node == "middle"
        assert result.response != ""
        assert result.edge_id == "e_start_mid"


# ---------------------------------------------------------------------------
# 7.2 - TurnResult on stay
# ---------------------------------------------------------------------------


class TestStayTurnResult:
    @pytest.mark.anyio
    async def test_stay_returns_turn_result(self) -> None:
        adapter = ControlAdapter()
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=False,
                reason="not ready",
                response="Tell me more about what you need",
            )
        )
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)

        result = await machine.process_turn("hmm")

        assert result.outcome == "stay"
        assert result.from_node == "start"
        assert result.to_node == "start"
        assert result.response == "Tell me more about what you need"


# ---------------------------------------------------------------------------
# 7.3 - TurnResult on error
# ---------------------------------------------------------------------------


class TestErrorTurnResult:
    @pytest.mark.anyio
    async def test_error_returns_turn_result(self) -> None:
        adapter = ErrorAdapter()
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)

        result = await machine.process_turn("anything")

        assert result.outcome == "error"
        assert result.from_node == "start"
        assert result.to_node == "start"
        assert "recovery:start" in result.response
        assert result.error is not None
        assert "LLM exploded" in result.error


# ---------------------------------------------------------------------------
# 7.4 - CriteriaResult.response used for stay guidance
# ---------------------------------------------------------------------------


class TestStayGuidance:
    @pytest.mark.anyio
    async def test_criteria_response_used(self) -> None:
        adapter = ControlAdapter()
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=False,
                response="What date works for you?",
            )
        )
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)

        result = await machine.process_turn("4 people")

        assert result.response == "What date works for you?"


# ---------------------------------------------------------------------------
# 7.5 - Fallback to generate_reply when response is empty
# ---------------------------------------------------------------------------


class TestStayFallback:
    @pytest.mark.anyio
    async def test_fallback_to_generate_reply(self) -> None:
        adapter = ControlAdapter()
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=False,
                response=None,
            )
        )
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)

        result = await machine.process_turn("hello")

        # Should fall back to generate_reply
        assert result.response == "reply:start"
        assert "reply:start" in adapter.replies


# ---------------------------------------------------------------------------
# 7.6 - Edge action execution
# ---------------------------------------------------------------------------


class TestEdgeActions:
    @pytest.mark.anyio
    async def test_edge_actions_fired(self) -> None:
        adapter = ControlAdapter()
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=True,
                recommended_edge_id="e_to_end",
            )
        )
        machine = await DialogStateMachine.from_flow(_flow_with_actions(), adapter)

        result = await machine.process_turn("go")

        assert result.outcome == "transition"
        assert "act_1" in result.actions_fired


# ---------------------------------------------------------------------------
# 7.7 - Node ON_ENTER action execution
# ---------------------------------------------------------------------------


class TestNodeActions:
    @pytest.mark.anyio
    async def test_on_enter_actions_fired(self) -> None:
        adapter = ControlAdapter()
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=True,
                recommended_edge_id="e_to_end",
            )
        )
        machine = await DialogStateMachine.from_flow(_flow_with_actions(), adapter)

        result = await machine.process_turn("go")

        assert "act_2" in result.actions_fired


# ---------------------------------------------------------------------------
# 7.8 - store_response_as writes to userdata
# ---------------------------------------------------------------------------


class TestStoreResponseAs:
    @pytest.mark.anyio
    async def test_action_result_stored(self) -> None:
        adapter = ControlAdapter()
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=True,
                recommended_edge_id="e_to_end",
            )
        )
        machine = await DialogStateMachine.from_flow(_flow_with_actions(), adapter)

        await machine.process_turn("go")

        assert "api_result" in machine.context.userdata
        assert machine.context.userdata["api_result"]["status"] == 200


# ---------------------------------------------------------------------------
# 7.9 - Action failure is non-fatal
# ---------------------------------------------------------------------------


class FailingActionAdapter(ControlAdapter):
    async def execute_action(
        self, action: CustomAction, userdata: dict[str, Any]
    ) -> dict[str, Any] | None:
        raise RuntimeError("action failed")


class TestActionFailure:
    @pytest.mark.anyio
    async def test_action_failure_non_fatal(self) -> None:
        adapter = FailingActionAdapter()
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=True,
                recommended_edge_id="e_to_end",
            )
        )
        machine = await DialogStateMachine.from_flow(_flow_with_actions(), adapter)

        result = await machine.process_turn("go")

        # Transition should still succeed despite action failure
        assert result.outcome == "transition"
        assert machine.current_state == "end"


# ---------------------------------------------------------------------------
# 7.10 - visit_count increments on re-entry
# ---------------------------------------------------------------------------


class TestVisitCount:
    @pytest.mark.anyio
    async def test_visit_count_increments(self) -> None:
        adapter = ControlAdapter()
        # start → middle
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=True,
                recommended_edge_id="e_to_mid",
            )
        )
        # middle → start (back)
        adapter.queue_result(
            CriteriaResult(
                node_id="middle",
                all_required_met=True,
                recommended_edge_id="e_back",
            )
        )
        machine = await DialogStateMachine.from_flow(_loop_flow(), adapter)

        assert machine.context.visit_count["start"] == 1

        await machine.process_turn("go")
        assert machine.context.visit_count["middle"] == 1

        await machine.process_turn("back")
        assert machine.context.visit_count["start"] == 2


# ---------------------------------------------------------------------------
# 7.11 - turns_in_node increments on stay, resets on transition
# ---------------------------------------------------------------------------


class TestTurnsInNode:
    @pytest.mark.anyio
    async def test_turns_increment_and_reset(self) -> None:
        adapter = ControlAdapter()
        # Two stays then a transition
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=False,
                response="stay 1",
            )
        )
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=False,
                response="stay 2",
            )
        )
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=True,
                recommended_edge_id="e_start_mid",
            )
        )
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)

        await machine.process_turn("a")
        assert machine.context.turns_in_node == 1

        await machine.process_turn("b")
        assert machine.context.turns_in_node == 2

        await machine.process_turn("c")
        # After transition, resets to 0
        assert machine.context.turns_in_node == 0


# ---------------------------------------------------------------------------
# 7.12 - max_turns triggers fallback edge
# ---------------------------------------------------------------------------


class TestMaxTurns:
    @pytest.mark.anyio
    async def test_max_turns_triggers_fallback(self) -> None:
        adapter = ControlAdapter()
        # Queue stays (will be ignored once max_turns exceeded)
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=False,
                response="try 1",
            )
        )
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=False,
                response="try 2",
            )
        )
        machine = await DialogStateMachine.from_flow(_flow_with_fallback(), adapter)

        # Turn 1, 2 are within max_turns=2
        await machine.process_turn("a")
        await machine.process_turn("b")
        assert machine.current_state == "start"

        # Turn 3 exceeds max_turns → fallback fires
        result = await machine.process_turn("c")
        assert result.outcome == "transition"
        assert machine.current_state == "end"


# ---------------------------------------------------------------------------
# 7.13 - InMemoryContextStore save/load/delete
# ---------------------------------------------------------------------------


class TestInMemoryContextStore:
    @pytest.mark.anyio
    async def test_save_load_roundtrip(self) -> None:
        from superdialog.machine.models import FlowContext

        store = InMemoryContextStore()
        ctx = FlowContext(
            current_node_id="test",
            userdata={"key": "value"},
        )

        await store.save("s1", ctx)
        loaded = await store.load("s1")

        assert loaded is not None
        assert loaded.current_node_id == "test"
        assert loaded.userdata["key"] == "value"

    @pytest.mark.anyio
    async def test_load_missing(self) -> None:
        store = InMemoryContextStore()
        assert await store.load("nonexistent") is None

    @pytest.mark.anyio
    async def test_delete(self) -> None:
        from superdialog.machine.models import FlowContext

        store = InMemoryContextStore()
        await store.save("s1", FlowContext(current_node_id="x"))
        await store.delete("s1")
        assert await store.load("s1") is None


# ---------------------------------------------------------------------------
# 7.14 - Fire-and-forget save does not block
# ---------------------------------------------------------------------------


class TestFireAndForgetSave:
    @pytest.mark.anyio
    async def test_save_runs_in_background(self) -> None:
        import asyncio

        store = InMemoryContextStore()
        adapter = ControlAdapter()
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=False,
                response="stay",
            )
        )
        machine = await DialogStateMachine.from_flow(
            _simple_flow(),
            adapter,
            session_id="sess1",
            store=store,
        )

        result = await machine.process_turn("hello")

        # Allow background task to complete
        await asyncio.sleep(0.05)

        saved = await store.load("sess1")
        assert saved is not None
        assert result.outcome == "stay"


# ---------------------------------------------------------------------------
# 7.15 - Context restore on from_flow with session_id
# ---------------------------------------------------------------------------


class TestContextRestore:
    @pytest.mark.anyio
    async def test_restore_from_store(self) -> None:
        from superdialog.machine.models import FlowContext

        store = InMemoryContextStore()
        saved_ctx = FlowContext(
            current_node_id="middle",
            conversation_history=[
                {"role": "user", "content": "previous"},
            ],
            visit_count={"start": 1, "middle": 1},
        )
        await store.save("sess1", saved_ctx)

        adapter = ControlAdapter()
        machine = await DialogStateMachine.from_flow(
            _simple_flow(),
            adapter,
            session_id="sess1",
            store=store,
        )

        assert machine.current_state == "middle"
        assert len(machine.context.conversation_history) == 1


# ---------------------------------------------------------------------------
# 7.16 - LLM retry on parse failure
# ---------------------------------------------------------------------------


class TestLLMRetry:
    @pytest.mark.anyio
    async def test_retry_on_parse_failure(self) -> None:
        from superdialog.machine.criteria import CriteriaJudge

        call_count = 0

        async def flaky_llm(messages: list[dict]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "not json at all"
            return (
                '{"criteria_met": {}, "all_required_met": false,'
                ' "user_insisting": false, "recommended_edge_id": null,'
                ' "reason": "retry worked", "response": "got it"}'
            )

        judge = CriteriaJudge(llm_fn=flaky_llm)
        result = await judge.evaluate(
            FlowNode(id="test", name="Test"),
            history=[],
            userdata={},
        )

        assert call_count == 2
        assert result.reason == "retry worked"
        assert result.response == "got it"


# ---------------------------------------------------------------------------
# 7.17 - Assistant message recorded in conversation_history
# ---------------------------------------------------------------------------


class TestHistoryRecording:
    @pytest.mark.anyio
    async def test_stay_records_assistant_message(self) -> None:
        adapter = ControlAdapter()
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=False,
                response="guidance text",
            )
        )
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)

        await machine.process_turn("hello")

        history = machine.context.conversation_history
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "guidance text"

    @pytest.mark.anyio
    async def test_transition_records_assistant_message(self) -> None:
        adapter = ControlAdapter()
        adapter.queue_result(
            CriteriaResult(
                node_id="start",
                all_required_met=True,
                recommended_edge_id="e_start_mid",
            )
        )
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)

        await machine.process_turn("go")

        history = machine.context.conversation_history
        # user msg + assistant response for new node
        assert any(m["role"] == "assistant" for m in history)

    @pytest.mark.anyio
    async def test_error_records_assistant_message(self) -> None:
        adapter = ErrorAdapter()
        machine = await DialogStateMachine.from_flow(_simple_flow(), adapter)

        await machine.process_turn("anything")

        history = machine.context.conversation_history
        assert len(history) == 2
        assert history[1]["role"] == "assistant"
        assert "recovery" in history[1]["content"]
