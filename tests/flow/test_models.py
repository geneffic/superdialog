"""Round-trip tests for ConversationFlow against real flow JSONs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superdialog.flow import Flow

FIXTURES = Path(__file__).parent.parent / "fixtures" / "flow"


@pytest.mark.parametrize(
    "name",
    [
        "kyc.json",
        "appointment.json",
        "escalation.json",
    ],
)
def test_existing_flow_roundtrip(name: str) -> None:
    raw = json.loads((FIXTURES / name).read_text())
    flow = Flow.model_validate(raw)
    redumped = flow.model_dump(exclude_unset=True, by_alias=True)
    assert redumped == raw, (
        f"roundtrip diff for {name}: "
        f"missing={set(raw) - set(redumped)} extra={set(redumped) - set(raw)}"
    )


def test_initial_node_preserved() -> None:
    raw = json.loads((FIXTURES / "kyc.json").read_text())
    flow = Flow.model_validate(raw)
    assert flow.initial_node == raw["initial_node"]
    assert len(flow.nodes) == len(raw["nodes"])


def test_tool_definition_with_discriminator_fields() -> None:
    """ToolDefinition placeholder accepts the new type/url/method/server."""
    raw = {
        "id": "search",
        "name": "search",
        "description": "Search the web",
        "type": "http",
        "method": "GET",
        "url": "https://example.com/search",
    }
    from superdialog.flow.models import ToolDefinition

    tool = ToolDefinition.model_validate(raw)
    assert tool.type == "http"
    assert tool.url == "https://example.com/search"
    assert tool.method == "GET"
    assert tool.server is None


def test_tool_definition_defaults_preserve_legacy() -> None:
    from superdialog.flow.models import ToolDefinition

    tool = ToolDefinition(id="t", name="t", description="d")
    assert tool.type == "python"
    assert tool.handler_id is None
    assert tool.input_schema is None
