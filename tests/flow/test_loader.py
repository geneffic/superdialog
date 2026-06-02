"""Tests for the Flow.save / Flow.load / FlowSet helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superdialog.flow import Flow, FlowSet

FIXTURES = Path(__file__).parent.parent / "fixtures" / "flow"

_MINIMAL_RAW: dict = {
    "system_prompt": "",
    "initial_node": "a",
    "nodes": [
        {
            "id": "a",
            "name": "A",
            "static_text": "hi",
            "is_final": True,
        }
    ],
}


@pytest.mark.parametrize(
    "fixture_name", ["kyc.json", "appointment.json", "escalation.json"]
)
def test_flow_save_load_roundtrip(tmp_path: Path, fixture_name: str) -> None:
    src = FIXTURES / fixture_name
    flow = Flow.load(src)
    out = tmp_path / "out.json"
    flow.save(out)
    reloaded = Flow.load(out)
    assert reloaded.initial_node == flow.initial_node
    assert len(reloaded.nodes) == len(flow.nodes)


def test_flowset_access() -> None:
    flow = Flow.model_validate(_MINIMAL_RAW)
    fs = FlowSet({"main": flow})
    assert "main" in fs.names()
    assert fs["main"].initial_node == "a"


def test_flowset_contains() -> None:
    fs = FlowSet({"main": Flow.model_validate(_MINIMAL_RAW)})
    assert "main" in fs
    assert "missing" not in fs


def test_load_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Flow.load(tmp_path / "does-not-exist.json")


def test_load_raises_on_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        Flow.load(bad)


def test_load_raises_on_missing_initial_node(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"system_prompt": "", "nodes": []}))
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Flow.load(bad)
