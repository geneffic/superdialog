"""superdialog CLI: chat / flow lint / flow draw / flow generate / playbook.

Each subcommand operates on a flow file (JSON or YAML) loaded via
:meth:`superdialog.Flow.load`. The CLI is intentionally thin -- it
defers all real work to the public API so ``superdialog flow lint X``
behaves the same as a Python caller would.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import yaml
from dotenv import load_dotenv

from .. import DialogMachine, Flow, create_dialog_flow


def _run_chat_repl(flow: "Flow", llm: str, adapter: str = "llm") -> None:
    """Blocking interactive REPL. Separated for testability."""
    machine = DialogMachine(flow=flow, llm=llm, adapter=adapter)

    async def _loop() -> None:
        result = await machine.start()
        if result.text:
            print(result.text)
        while True:
            try:
                user = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if user.strip() in {"quit", "exit"}:
                return
            if not user.strip():
                continue
            _t0 = time.monotonic()
            turn = await machine.turn(user)
            _ms = int((time.monotonic() - _t0) * 1000)
            if turn.text:
                print(turn.text)
            print(f"[{_ms}ms]", file=sys.stderr)
            if machine.is_complete:
                return

    asyncio.run(_loop())


def _cmd_chat(args: argparse.Namespace) -> int:
    """Interactive REPL: auto-detect flow.json in cwd or use --flow path."""
    load_dotenv()

    flow_path = getattr(args, "flow", "flow.json") or "flow.json"
    if not Path(flow_path).exists():
        print(
            f"No flow found at: {flow_path}\n"
            f"Run: superdialog flow generate --output {flow_path}",
            file=sys.stderr,
        )
        return 1

    flow = Flow.load(flow_path)
    llm = getattr(args, "llm", "openai/gpt-4o-mini") or "openai/gpt-4o-mini"
    adapter = getattr(args, "adapter", "llm") or "llm"
    _run_chat_repl(flow, llm, adapter)
    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    """Validate edge target references; exit non-zero on broken refs."""
    flow = Flow.load(args.flow)
    issues = _lint_flow(flow)
    if not issues:
        print("OK")
        return 0
    for issue in issues:
        print(issue)
    return 1


def _lint_flow(flow: Any) -> list[str]:
    """Return a list of human-readable issues (empty list = clean flow)."""
    issues: list[str] = []
    node_ids = {n.id for n in flow.nodes}
    for node in flow.nodes:
        for edge in node.edges or []:
            target = edge.target_node_id
            if target and target not in node_ids:
                issues.append(
                    f"node {node.id!r}: edge {edge.id!r} -> unknown target {target!r}"
                )
    for gedge in getattr(flow, "global_edges", []) or []:
        target = gedge.target_node_id
        if target and target not in node_ids:
            issues.append(f"global edge {gedge.id!r} -> unknown target {target!r}")

    # Warn when a required criteria key is not found in any edge input_schema property
    for node in flow.nodes:
        criteria = getattr(node, "completion_criteria", None) or []
        all_edge_schema_keys: set[str] = set()
        for edge in node.edges or []:
            schema = getattr(edge, "input_schema", None)
            if isinstance(schema, dict):
                props = schema.get("properties", {})
                if isinstance(props, dict):
                    all_edge_schema_keys.update(props.keys())
        for criterion in criteria:
            required = getattr(criterion, "required", True)
            if not required:
                continue
            key = getattr(criterion, "key", None)
            if key and all_edge_schema_keys and key not in all_edge_schema_keys:
                issues.append(
                    f"node {node.id!r}: criteria key {key!r} is required but not found "
                    f"in any edge input_schema - the LLM may never extract this value"
                )

    return issues


def _cmd_draw(args: argparse.Namespace) -> int:
    """Emit a Mermaid ``graph TD`` rendering of the flow's edges."""
    flow = Flow.load(args.flow)
    for line in _draw_mermaid(flow):
        print(line)
    return 0


def _draw_mermaid(flow: Any) -> list[str]:
    lines = ["graph TD"]
    for node in flow.nodes:
        for edge in node.edges or []:
            if edge.target_node_id:
                lines.append(f"  {node.id} -->|{edge.id}| {edge.target_node_id}")
    for gedge in getattr(flow, "global_edges", []) or []:
        if gedge.target_node_id:
            lines.append(f"  * -->|{gedge.id}| {gedge.target_node_id}")
    return lines


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate a flow JSON from a natural-language prompt or description file."""
    load_dotenv()

    # Resolve description text
    from_file = getattr(args, "from_file", None)
    if from_file and getattr(args, "prompt", None):
        print("Warning: --from provided; ignoring positional prompt", file=sys.stderr)
    if from_file:
        p = Path(from_file)
        if not p.exists():
            print(f"Error: description file not found: {from_file}", file=sys.stderr)
            return 1
        prompt = p.read_text()
    else:
        prompt = getattr(args, "prompt", None)

    if not prompt or not prompt.strip():
        print("Error: provide a prompt or --from <file>", file=sys.stderr)
        return 1

    output = getattr(args, "output", "flow.json") or "flow.json"
    llm = getattr(args, "llm", "openai/gpt-4o-mini") or "openai/gpt-4o-mini"

    print(f"Generating flow using {llm}...", flush=True)
    flow = asyncio.run(create_dialog_flow(prompt=prompt.strip(), llm=llm))
    flow.save(output)

    node_count = len(flow.nodes)
    edge_count = sum(len(n.edges or []) for n in flow.nodes)
    print(f"Saved: {output}  ({node_count} nodes, {edge_count} edges)")

    # Auto-lint: run checks immediately after generation
    issues = _lint_flow(flow)
    if issues:
        print(f"Lint warnings ({len(issues)}):")
        for issue in issues:
            print(f"  warning: {issue}")
        print(f"Run 'superdialog flow lint {output}' to re-check after edits.")
    else:
        print("Lint: OK")

    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Run eval against a flow — audit mode (with --traversal) or synthetic mode."""
    load_dotenv()
    from superdialog.machine.eval import run_eval
    from superdialog.machine.eval.models import AuditReport, EvalReport

    traversal = getattr(args, "traversal", None)
    model = getattr(args, "model", "gpt-4.1-mini") or "gpt-4.1-mini"

    print(f"Running eval: flow={args.flow} model={model}", file=sys.stderr)
    if traversal:
        print(f"Mode: audit  traversal={traversal}", file=sys.stderr)
    else:
        print("Mode: synthetic (generating corpus)", file=sys.stderr)

    report = asyncio.run(run_eval(
        flow_path=args.flow,
        traversal_path=traversal,
        model=model,
    ))

    if isinstance(report, AuditReport):
        print(report.to_markdown())
        return 0 if report.overall_score >= 0.0 else 1

    # EvalReport
    for score in report.models:
        print(f"\n{'='*50}")
        print(f"Model:              {score.model_id}")
        print(f"Edge accuracy:      {score.edge_accuracy:.1%}")
        print(f"Persona completion: {score.persona_completion:.1%}")
        if score.failures:
            print(f"\nFailures ({len(score.failures)}):")
            for f in score.failures:
                print(f"  - {f}")
    return 0


def _compile_flow_to_yaml(flow_path: str) -> str:
    """Compile a ConversationFlow JSON to Playbook YAML string."""
    from superdialog.flow.models import ConversationFlow
    from superdialog.playbook import compile_flow

    flow = ConversationFlow.load(flow_path)
    pb = compile_flow(flow)
    return yaml.safe_dump(
        pb.model_dump(mode="json", exclude_defaults=True),
        sort_keys=False,
        allow_unicode=True,
        width=80,
    )


def _cmd_playbook_compile(args: argparse.Namespace) -> int:
    """Compile a flow JSON to Playbook YAML; emit to stdout or --output file."""
    text = _compile_flow_to_yaml(args.flow)
    output = getattr(args, "output", None)
    if output:
        Path(output).write_text(text)
        print(f"Written: {output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


async def _playbook_chat_loop(playbook_path: str, model: str) -> None:
    import anyio

    from superdialog.llm.litellm_provider import LitellmProvider
    from superdialog.playbook import Playbook, httpx_http
    from superdialog.playbook.director import Director
    from superdialog.playbook.runtime import PlaybookRuntime
    from superdialog.playbook.talker import Talker

    class _DirectorLLM:
        def __init__(self, provider: LitellmProvider) -> None:
            self._p = provider

        async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
            return (await self._p.complete(list(messages))).text

    class _TalkerLLM:
        def __init__(self, provider: LitellmProvider) -> None:
            self._p = provider

        async def stream(
            self, messages: list[dict[str, str]], **kw: Any
        ) -> AsyncIterator[str]:
            async for chunk in self._p.stream(list(messages)):
                if chunk.text:
                    yield chunk.text

    provider = LitellmProvider(model)
    playbook = Playbook.load(playbook_path)
    runtime = PlaybookRuntime(
        playbook, director_llm=_DirectorLLM(provider), http=httpx_http
    )
    talker = Talker(playbook, llm=_TalkerLLM(provider))

    async def _speak() -> None:
        print("agent> ", end="", flush=True)
        async for chunk in talker.speak(runtime.state):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()

    print(f"Playbook demo on {model} — type 'quit' to exit.\n")
    await runtime.start()
    await _speak()

    while True:
        try:
            text = await anyio.to_thread.run_sync(input, "you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in {"quit", "exit"}:
            break
        if not text.strip():
            continue
        cp_before = runtime.state.checkpoint_id
        await runtime.on_user_text(text)
        cp_after = runtime.state.checkpoint_id
        print(f"[DEBUG] {cp_before} → {cp_after}  ended={runtime.state.ended}",
              file=sys.stderr)
        await _speak()
        if runtime.state.ended:
            print(f"[session ended — outcome: {runtime.state.outcome}]")
            break


def _cmd_playbook_chat(args: argparse.Namespace) -> int:
    """Run the playbook REPL against an existing YAML file."""
    load_dotenv()
    playbook_path = args.playbook
    if not Path(playbook_path).exists():
        print(f"Playbook not found: {playbook_path}", file=sys.stderr)
        return 1
    model = getattr(args, "model", "openai/gpt-4.1-mini") or "openai/gpt-4.1-mini"
    import anyio
    anyio.run(_playbook_chat_loop, playbook_path, model)
    return 0


def _cmd_playbook_run(args: argparse.Namespace) -> int:
    """Compile a flow JSON to a playbook YAML, then start the REPL."""
    load_dotenv()
    import tempfile
    model = getattr(args, "model", "openai/gpt-4.1-mini") or "openai/gpt-4.1-mini"

    # If arg is already a .yaml/.yml file, skip compile
    src = args.flow
    if src.endswith((".yaml", ".yml")):
        playbook_path = src
    else:
        text = _compile_flow_to_yaml(src)
        output = getattr(args, "output", None)
        if output:
            Path(output).write_text(text)
            playbook_path = output
            print(f"Compiled → {output}", file=sys.stderr)
        else:
            # Write to a temp file so the REPL can load it
            tmp = tempfile.NamedTemporaryFile(
                suffix=".yaml", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(text)
            tmp.close()
            playbook_path = tmp.name
            print(f"Compiled → {playbook_path} (temp)", file=sys.stderr)

    import anyio
    anyio.run(_playbook_chat_loop, playbook_path, model)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="superdialog")
    sub = parser.add_subparsers(dest="cmd", required=True)

    chat = sub.add_parser("chat", help="Interactive REPL against a flow")
    chat.add_argument(
        "--flow",
        default="flow.json",
        help="Path to flow JSON (default: ./flow.json)",
    )
    chat.add_argument("--llm", default="openai/gpt-4o-mini")
    chat.add_argument(
        "--adapter",
        default="toolcall",
        choices=["llm", "toolcall"],
        help="Adapter mode: 'toolcall' (default, 1 LLM call/turn, mirrors production) or "
        "'llm' (2 LLM calls/turn)",
    )
    chat.set_defaults(fn=_cmd_chat)

    flow = sub.add_parser("flow", help="Inspect / manipulate flow files")
    flow_sub = flow.add_subparsers(dest="subcmd", required=True)

    lint = flow_sub.add_parser("lint", help="Validate edge target references")
    lint.add_argument("flow")
    lint.set_defaults(fn=_cmd_lint)

    draw = flow_sub.add_parser("draw", help="Print a Mermaid graph of the flow")
    draw.add_argument("flow")
    draw.set_defaults(fn=_cmd_draw)

    generate = flow_sub.add_parser(
        "generate", help="Generate a flow JSON from a natural-language prompt"
    )
    generate.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Inline description string (omit if using --from)",
    )
    generate.add_argument(
        "--from",
        dest="from_file",
        metavar="FILE",
        help="Path to description file (alternative to positional prompt)",
    )
    generate.add_argument(
        "--output",
        default="flow.json",
        help="Output path for flow JSON (default: flow.json)",
    )
    generate.add_argument("--llm", default="openai/gpt-4o-mini")
    generate.set_defaults(fn=_cmd_generate)

    # -- playbook subcommand group ------------------------------------------
    pb = sub.add_parser("playbook", help="Compile and run Playbook engine sessions")
    pb_sub = pb.add_subparsers(dest="pb_subcmd", required=True)

    pb_compile = pb_sub.add_parser(
        "compile", help="Compile a flow JSON to Playbook YAML"
    )
    pb_compile.add_argument("flow", help="Path to flow JSON")
    pb_compile.add_argument(
        "--output", "-o", default=None, help="Output YAML file (default: stdout)"
    )
    pb_compile.set_defaults(fn=_cmd_playbook_compile)

    pb_chat = pb_sub.add_parser(
        "chat", help="Interactive REPL against an existing Playbook YAML"
    )
    pb_chat.add_argument("--playbook", required=True, help="Path to playbook YAML")
    pb_chat.add_argument(
        "--model", default="openai/gpt-4.1-mini", help="LiteLLM model string"
    )
    pb_chat.set_defaults(fn=_cmd_playbook_chat)

    pb_run = pb_sub.add_parser(
        "run", help="Compile a flow JSON and immediately start a Playbook REPL"
    )
    pb_run.add_argument("flow", help="Path to flow JSON (or existing .yaml to skip compile)")
    pb_run.add_argument(
        "--output", "-o", default=None, help="Save compiled YAML here (default: temp file)"
    )
    pb_run.add_argument(
        "--model", default="openai/gpt-4.1-mini", help="LiteLLM model string"
    )
    pb_run.set_defaults(fn=_cmd_playbook_run)

    # -- eval subcommand ----------------------------------------------------
    eval_cmd = sub.add_parser(
        "eval",
        help="Eval a flow: audit real session (--traversal) or run synthetic corpus",
    )
    eval_cmd.add_argument("--flow", required=True, help="Path to flow JSON")
    eval_cmd.add_argument(
        "--traversal",
        default=None,
        help="Path to traversal JSON (audit mode). Omit for synthetic eval.",
    )
    eval_cmd.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model for eval LLM (default: gpt-4.1-mini)",
    )
    eval_cmd.set_defaults(fn=_cmd_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    rc = args.fn(args)
    return int(rc or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
