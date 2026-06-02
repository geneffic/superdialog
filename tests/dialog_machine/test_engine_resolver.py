"""Tests for super.core.voice.dialog_machine.engine."""

from __future__ import annotations

import pytest

from superdialog.machine.engine import (
    FlowEngine,
    UnknownFlowEngineError,
    resolve_flow_engine,
    resolve_from_config,
)


class TestResolveFlowEngine:
    def test_canonical_state_machine(self) -> None:
        assert resolve_flow_engine("state_machine") == FlowEngine.STATE_MACHINE

    def test_canonical_graph_engine(self) -> None:
        assert resolve_flow_engine("graph_engine") == FlowEngine.GRAPH_ENGINE

    def test_alias_langgraph_normalizes_to_state_machine(self) -> None:
        assert resolve_flow_engine("langgraph") == FlowEngine.STATE_MACHINE

    def test_case_insensitive(self) -> None:
        assert resolve_flow_engine("STATE_MACHINE") == FlowEngine.STATE_MACHINE
        assert resolve_flow_engine("LangGraph") == FlowEngine.STATE_MACHINE

    def test_strips_whitespace(self) -> None:
        assert resolve_flow_engine("  state_machine  ") == FlowEngine.STATE_MACHINE

    def test_empty_string_defaults_to_state_machine(self) -> None:
        assert resolve_flow_engine("") == FlowEngine.STATE_MACHINE

    def test_none_defaults_to_state_machine(self) -> None:
        assert resolve_flow_engine(None) == FlowEngine.STATE_MACHINE

    def test_unknown_value_raises(self) -> None:
        with pytest.raises(UnknownFlowEngineError) as exc_info:
            resolve_flow_engine("garbage")
        msg = str(exc_info.value)
        assert "garbage" in msg
        assert "state_machine" in msg
        assert "graph_engine" in msg

    def test_typo_raises_loudly(self) -> None:
        # The previous bug: a typo silently fell back to the old FlowAgent.
        with pytest.raises(UnknownFlowEngineError):
            resolve_flow_engine("stat_machine")


class TestResolveFromConfig:
    def test_config_takes_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLOW_ENGINE", "graph_engine")
        result = resolve_from_config({"flow_engine": "state_machine"})
        assert result == FlowEngine.STATE_MACHINE

    def test_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FLOW_ENGINE", "graph_engine")
        result = resolve_from_config({})
        assert result == FlowEngine.GRAPH_ENGINE

    def test_alias_in_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FLOW_ENGINE", "langgraph")
        assert resolve_from_config({}) == FlowEngine.STATE_MACHINE

    def test_no_config_no_env_defaults_to_state_machine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FLOW_ENGINE", raising=False)
        assert resolve_from_config({}) == FlowEngine.STATE_MACHINE
        assert resolve_from_config(None) == FlowEngine.STATE_MACHINE

    def test_unknown_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FLOW_ENGINE", "garbage")
        with pytest.raises(UnknownFlowEngineError):
            resolve_from_config({})
