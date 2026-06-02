"""CLI tests for ``superdialog.cli.main``.

The ``chat`` subcommand drives a live LLM so we don't exercise it here;
``flow lint`` and ``flow draw`` are pure flow-file operations and can be
verified end-to-end against the bundled fixtures.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superdialog import Flow
from superdialog.cli.main import _build_parser, _lint_flow, main

_cli_main_module = importlib.import_module("superdialog.cli.main")

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "flow"


def test_flow_lint_clean_fixture(capsys: pytest.CaptureFixture) -> None:
    rc = main(["flow", "lint", str(FIXTURES / "kyc.json")])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == "OK"


def test_flow_lint_reports_broken_edge(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    flow = {
        "id": "broken",
        "system_prompt": "test",
        "initial_node": "a",
        "nodes": [
            {
                "id": "a",
                "name": "A",
                "edges": [
                    {
                        "id": "to_nowhere",
                        "condition": "always",
                        "target_node_id": "missing",
                    }
                ],
            }
        ],
    }
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(flow))

    rc = main(["flow", "lint", str(path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "unknown target 'missing'" in out


def test_flow_draw_emits_mermaid(capsys: pytest.CaptureFixture) -> None:
    rc = main(["flow", "draw", str(FIXTURES / "kyc.json")])
    out = capsys.readouterr().out
    assert rc == 0
    lines = out.strip().splitlines()
    assert lines[0] == "graph TD"
    assert any("-->" in line for line in lines[1:])


def test_parser_requires_subcommand(capsys: pytest.CaptureFixture) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_flow_draw_handles_appointment_fixture(
    capsys: pytest.CaptureFixture,
) -> None:
    rc = main(["flow", "draw", str(FIXTURES / "appointment.json")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "graph TD" in out


def _make_mock_flow(num_nodes=2):
    mock = MagicMock()
    mock.nodes = [MagicMock(edges=[MagicMock()]) for _ in range(num_nodes)]
    return mock


def test_flow_generate_saves_to_output_file(tmp_path, capsys):
    mock_flow = _make_mock_flow()
    out = tmp_path / "flow.json"

    with (
        patch.object(
            _cli_main_module, "create_dialog_flow", new_callable=MagicMock
        ) as mock_create,
        patch("asyncio.run", return_value=mock_flow),
    ):
        rc = main(["flow", "generate", "A booking agent", "--output", str(out)])

    assert rc == 0
    mock_create.assert_called_once()
    mock_flow.save.assert_called_once_with(str(out))
    assert "Saved" in capsys.readouterr().out


def test_flow_generate_from_file(tmp_path, capsys):
    desc = tmp_path / "desc.txt"
    desc.write_text("A golf tee-time booking agent")
    mock_flow = _make_mock_flow()
    out = tmp_path / "flow.json"

    with (
        patch.object(
            _cli_main_module, "create_dialog_flow", new_callable=MagicMock
        ) as mock_create,
        patch("asyncio.run", return_value=mock_flow),
    ):
        rc = main(["flow", "generate", "--from", str(desc), "--output", str(out)])

    assert rc == 0
    mock_create.assert_called_once()
    mock_flow.save.assert_called_once_with(str(out))


def test_flow_generate_from_file_not_found(capsys):
    rc = main(["flow", "generate", "--from", "/nonexistent/desc.txt"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_flow_generate_default_output_is_flow_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    mock_flow = _make_mock_flow()

    with (
        patch.object(
            _cli_main_module, "create_dialog_flow", new_callable=MagicMock
        ) as mock_create,
        patch("asyncio.run", return_value=mock_flow),
    ):
        rc = main(["flow", "generate", "A simple agent"])

    assert rc == 0
    mock_flow.save.assert_called_once_with("flow.json")


def test_chat_auto_detects_flow_json(tmp_path, monkeypatch):
    flow_data = {
        "id": "t",
        "system_prompt": "s",
        "initial_node": "n",
        "nodes": [{"id": "n", "name": "N", "edges": [], "is_final": True}],
    }
    (tmp_path / "flow.json").write_text(json.dumps(flow_data))
    monkeypatch.chdir(tmp_path)

    with patch.object(_cli_main_module, "_run_chat_repl") as mock_repl:
        rc = main(["chat"])

    assert rc == 0
    mock_repl.assert_called_once()
    call_args = mock_repl.call_args
    from superdialog import Flow

    assert isinstance(call_args[0][0], Flow)
    assert isinstance(call_args[0][1], str)


def test_chat_explicit_flow_path(tmp_path):
    flow_data = {
        "id": "t",
        "system_prompt": "s",
        "initial_node": "n",
        "nodes": [{"id": "n", "name": "N", "edges": [], "is_final": True}],
    }
    flow_file = tmp_path / "custom.json"
    flow_file.write_text(json.dumps(flow_data))

    with patch.object(_cli_main_module, "_run_chat_repl") as mock_repl:
        rc = main(["chat", "--flow", str(flow_file)])

    assert rc == 0
    mock_repl.assert_called_once()
    call_args = mock_repl.call_args
    from superdialog import Flow

    assert isinstance(call_args[0][0], Flow)
    assert isinstance(call_args[0][1], str)


def test_chat_missing_flow_returns_1_with_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = main(["chat"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "generate" in err.lower()


def _make_flow_file(tmp_path: Path, flow_data: dict) -> str:
    """Helper: write flow_data to a temp JSON file and return the path."""
    import json

    flow_file = tmp_path / "test_flow.json"
    flow_file.write_text(json.dumps(flow_data))
    return str(flow_file)


def test_lint_warns_when_criteria_key_not_in_any_edge_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Warn when a criteria key is never referenced in any edge input_schema."""
    flow_data = {
        "id": "test",
        "system_prompt": "test",
        "initial_node": "collect",
        "nodes": [
            {
                "id": "collect",
                "name": "Collect",
                "instruction": "Ask for email.",
                "completion_criteria": [
                    {"key": "email", "description": "User email", "required": True}
                ],
                "edges": [
                    {
                        "id": "phone_given",
                        "condition": "User gave phone number",
                        "target_node_id": "done",
                        "input_schema": {
                            "type": "object",
                            "properties": {"phone": {"type": "string"}},
                        },
                    }
                ],
            },
            {"id": "done", "name": "Done", "edges": []},
        ],
    }
    flow = Flow.load(_make_flow_file(tmp_path, flow_data))
    issues = _lint_flow(flow)
    assert any("email" in i and "criteria" in i.lower() for i in issues)


def test_lint_no_warn_when_criteria_key_in_edge_schema(
    tmp_path: Path,
) -> None:
    """No warning when required criteria key matches an edge input_schema property."""
    flow_data = {
        "id": "test",
        "system_prompt": "test",
        "initial_node": "collect",
        "nodes": [
            {
                "id": "collect",
                "name": "Collect",
                "instruction": "Ask for email.",
                "completion_criteria": [
                    {"key": "email", "description": "User email", "required": True}
                ],
                "edges": [
                    {
                        "id": "email_given",
                        "condition": "User gave email",
                        "target_node_id": "done",
                        "input_schema": {
                            "type": "object",
                            "properties": {"email": {"type": "string"}},
                        },
                    }
                ],
            },
            {"id": "done", "name": "Done", "edges": []},
        ],
    }
    flow = Flow.load(_make_flow_file(tmp_path, flow_data))
    issues = _lint_flow(flow)
    criteria_issues = [i for i in issues if "criteria" in i.lower() and "email" in i]
    assert criteria_issues == []


def test_lint_no_warn_for_optional_criteria_not_in_edge_schema(
    tmp_path: Path,
) -> None:
    """No warning for optional criteria keys not found in edge schemas."""
    flow_data = {
        "id": "test",
        "system_prompt": "test",
        "initial_node": "collect",
        "nodes": [
            {
                "id": "collect",
                "name": "Collect",
                "instruction": "Ask for email.",
                "completion_criteria": [
                    {"key": "email", "description": "User email", "required": False}
                ],
                "edges": [
                    {
                        "id": "done_edge",
                        "condition": "User is done",
                        "target_node_id": "done",
                        "input_schema": {
                            "type": "object",
                            "properties": {"other": {"type": "string"}},
                        },
                    }
                ],
            },
            {"id": "done", "name": "Done", "edges": []},
        ],
    }
    flow = Flow.load(_make_flow_file(tmp_path, flow_data))
    issues = _lint_flow(flow)
    criteria_issues = [i for i in issues if "criteria" in i.lower()]
    assert criteria_issues == []


def test_generate_runs_lint_and_prints_issues(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """flow generate should run lint and print any issues found."""
    from unittest.mock import MagicMock, patch

    broken_node = MagicMock()
    broken_node.id = "node_a"
    broken_node.name = "Node A"
    broken_node.completion_criteria = []
    broken_edge = MagicMock()
    broken_edge.id = "edge_1"
    broken_edge.target_node_id = "missing_node"
    broken_edge.input_schema = None
    broken_node.edges = [broken_edge]

    broken_flow = MagicMock()
    broken_flow.nodes = [broken_node]
    broken_flow.global_edges = []

    output_file = str(tmp_path / "flow.json")
    broken_flow.save = MagicMock()

    with (
        patch("superdialog.cli.main.create_dialog_flow", return_value=broken_flow),
        patch("superdialog.cli.main.asyncio") as mock_asyncio,
    ):
        mock_asyncio.run.return_value = broken_flow
        rc = main(["flow", "generate", "test prompt", "--output", output_file])

    out = capsys.readouterr().out
    assert "Lint warnings" in out
    assert "warning: " in out
    assert "missing_node" in out
    assert "superdialog flow lint" in out


def test_generate_prints_lint_ok_when_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """flow generate should print 'Lint: OK' when no issues found."""
    from unittest.mock import MagicMock, patch

    clean_node = MagicMock()
    clean_node.id = "node_a"
    clean_node.name = "Node A"
    clean_node.completion_criteria = []
    clean_edge = MagicMock()
    clean_edge.id = "edge_1"
    clean_edge.target_node_id = "done"
    clean_edge.input_schema = None
    clean_node.edges = [clean_edge]

    done_node = MagicMock()
    done_node.id = "done"
    done_node.name = "Done"
    done_node.completion_criteria = []
    done_node.edges = []

    clean_flow = MagicMock()
    clean_flow.nodes = [clean_node, done_node]
    clean_flow.global_edges = []

    output_file = str(tmp_path / "flow.json")
    clean_flow.save = MagicMock()

    with (
        patch("superdialog.cli.main.create_dialog_flow", return_value=clean_flow),
        patch("superdialog.cli.main.asyncio") as mock_asyncio,
    ):
        mock_asyncio.run.return_value = clean_flow
        rc = main(["flow", "generate", "test prompt", "--output", output_file])

    out = capsys.readouterr().out
    assert "Lint: OK" in out
