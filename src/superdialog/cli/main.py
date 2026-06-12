"""superdialog CLI: chat / flow lint / flow draw / flow generate.

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
from typing import Any, AsyncIterator, Sequence, cast

import yaml
from dotenv import load_dotenv

from .. import DialogMachine, Flow, create_dialog_flow
from ..stream import StreamChunk


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


def _looks_like_playbook(path: str) -> bool:
    """True when ``path`` parses to a mapping carrying a top-level 'journeys'.

    Tolerant by design: any read/parse failure means "not a playbook" so the
    caller falls back to the flow loader, which then reports the real error.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(doc, dict) and "journeys" in doc


def _looks_like_simple_playbook(path: str) -> bool:
    """True when ``path`` parses to a simple playbook (top-level ``playbook`` list).

    Tolerant by design: any read/parse failure means "not a simple playbook" so
    the caller falls back to the flow loader, which then reports the real error.
    """
    from ..playbook.simple import is_simple_playbook

    try:
        text = Path(path).read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
    except (OSError, yaml.YAMLError):
        return False
    return is_simple_playbook(doc)


def _drive_agent(agent: Any) -> None:
    """Run the shared async REPL loop over a PlaybookAgent."""

    async def _loop() -> None:
        for line in await agent.runtime.start():
            print(line)
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
            chunks = cast(
                AsyncIterator[StreamChunk], await agent.turn(user, stream=True)
            )
            async for chunk in chunks:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
            state = agent.runtime.state
            print(
                f"[checkpoint={state.checkpoint_id} ended={state.ended}]",
                file=sys.stderr,
            )
            if state.ended:
                return

    asyncio.run(_loop())


def _build_playbook_agent(playbook: Any, llm: str) -> Any:
    """Build a PlaybookAgent for ``playbook`` using a single resolved model."""
    from ..llm.resolver import resolve_llm
    from ..playbook import PlaybookAgent, httpx_http, provider_adapters

    provider = resolve_llm(llm)
    director, talker = provider_adapters(provider)
    return PlaybookAgent(
        playbook=playbook,
        talker_llm=talker,
        director_llm=director,
        http=httpx_http,
    )


def _run_playbook_repl(playbook_path: str, llm: str) -> None:
    """Blocking interactive REPL driving a Playbook. Separated for testability.

    One model drives both the Director and the Talker here — fine for a dev
    REPL. Production splits them (a cheap streaming Talker, a stronger
    Director); see :func:`superdialog.playbook.provider_adapters`.
    """
    from ..playbook import Playbook

    agent = _build_playbook_agent(Playbook.load(playbook_path), llm)
    _drive_agent(agent)


def _run_simple_repl(simple_path: str, llm: str) -> None:
    """Blocking interactive REPL driving a compiled simple playbook."""
    from ..playbook.simple import load_simple

    agent = _build_playbook_agent(load_simple(simple_path), llm)
    _drive_agent(agent)


def _cmd_chat(args: argparse.Namespace) -> int:
    """Interactive REPL over a flow or a playbook (auto-detected or explicit)."""
    load_dotenv()

    llm = getattr(args, "llm", "openai/gpt-4o-mini") or "openai/gpt-4o-mini"

    playbook_path = getattr(args, "playbook", None)
    if playbook_path:
        if not Path(playbook_path).exists():
            print(f"No playbook found at: {playbook_path}", file=sys.stderr)
            return 1
        return _chat_playbook(playbook_path, llm)

    simple_path = getattr(args, "simple", None)
    if simple_path:
        if not Path(simple_path).exists():
            print(f"No simple playbook found at: {simple_path}", file=sys.stderr)
            return 1
        return _chat_simple(simple_path, llm)

    flow_path = getattr(args, "flow", "flow.json") or "flow.json"
    if not Path(flow_path).exists():
        print(
            f"No flow found at: {flow_path}\n"
            f"Run: superdialog flow generate --output {flow_path}",
            file=sys.stderr,
        )
        return 1

    if _looks_like_playbook(flow_path):
        return _chat_playbook(flow_path, llm)

    if _looks_like_simple_playbook(flow_path):
        return _chat_simple(flow_path, llm)

    try:
        flow = Flow.load(flow_path)
    except Exception as exc:  # malformed JSON / schema -> clean exit, no traceback
        print(f"Could not load flow {flow_path}: {exc}", file=sys.stderr)
        return 1
    adapter = getattr(args, "adapter", "llm") or "llm"
    _run_chat_repl(flow, llm, adapter)
    return 0


def _chat_playbook(path: str, llm: str) -> int:
    """Validate-load a playbook (clean error on failure) then run its REPL."""
    from ..playbook import Playbook

    try:
        Playbook.load(path)  # pre-flight: surface schema errors as one line
    except Exception as exc:
        print(f"Invalid playbook {path}: {exc}", file=sys.stderr)
        return 1
    _run_playbook_repl(path, llm)
    return 0


def _chat_simple(path: str, llm: str) -> int:
    """Validate-compile a simple playbook (clean error on failure) then run it."""
    from ..playbook.simple import load_simple

    try:
        load_simple(path)  # pre-flight: surface compile errors as one line
    except Exception as exc:
        print(f"Invalid simple playbook {path}: {exc}", file=sys.stderr)
        return 1
    _run_simple_repl(path, llm)
    return 0


def _run_optimize(
    playbook_path: str,
    *,
    rounds: int,
    n: int,
    personas_path: str | None,
    llm: str,
    candidate_llm: str | None,
    user_llm: str | None,
) -> tuple[str, list[str]]:
    """Run the optimize loop against real providers; return (yaml, trace lines)."""
    from ..llm.resolver import resolve_llm
    from ..playbook import (
        Playbook,
        PlaybookAgent,
        httpx_http,
        make_editable,
        optimize,
        provider_adapters,
    )
    from ..playbook.personas import (
        derive_default_persona,
        generate_personas,
        load_personas,
        persona_cache_path,
        save_personas,
    )

    doc = make_editable(Path(playbook_path).read_text(encoding="utf-8"))
    director, talker = provider_adapters(resolve_llm(llm))
    cand = (
        provider_adapters(resolve_llm(candidate_llm))[0] if candidate_llm else director
    )
    user = provider_adapters(resolve_llm(user_llm))[0] if user_llm else director

    def agent_factory(pb: "Playbook") -> "PlaybookAgent":
        return PlaybookAgent(
            playbook=pb, talker_llm=talker, director_llm=director, http=httpx_http
        )

    notes: list[str] = []

    async def _go() -> Any:
        playbook = doc.compile()
        cache = persona_cache_path(playbook_path)
        if personas_path:
            personas = load_personas(personas_path)
        elif Path(cache).exists():
            personas = load_personas(cache)
            notes.append(f"personas: loaded cache {cache}")
        else:
            try:
                personas = await generate_personas(playbook, cand)
                save_personas(personas, cache)
                notes.append(f"personas: generated suite -> {cache} (review it)")
            except ValueError as exc:
                personas = [derive_default_persona(playbook)]
                notes.append(
                    f"personas: generation failed ({exc}); using one derived persona"
                )
        return await optimize(
            doc,
            personas=personas,
            candidate_llm=cand,
            user_llm=user,
            agent_factory=agent_factory,
            rounds=rounds,
            n=n,
        )

    report = asyncio.run(_go())
    lines = list(notes)
    for t in report.trace:
        if t.candidate_breakdown is None:
            lines.append(f"round {t.round_no}: {t.detail}")
        else:
            n_edits = len(t.edits)
            verdict = (
                f"accepted ({n_edits} edit{'s' if n_edits != 1 else ''})"
                if t.accepted
                else "rejected"
            )
            lines.append(
                f"round {t.round_no}: incumbent "
                f"{t.incumbent_breakdown.objective:.2f} vs candidate "
                f"{t.candidate_breakdown.objective:.2f} - {verdict}"
            )
    lines.append(
        f"objective: {report.initial_breakdown.objective:.2f} -> "
        f"{report.final_breakdown.objective:.2f}"
    )
    return report.final_yaml, lines


def _cmd_optimize(args: argparse.Namespace) -> int:
    """Validate inputs, run the loop, write the improved playbook."""
    path = args.playbook
    if not Path(path).exists():
        print(f"Playbook not found: {path}", file=sys.stderr)
        return 1
    try:  # pre-flight either format; surface schema errors as one line
        if _looks_like_simple_playbook(path):
            from ..playbook.simple import load_simple

            load_simple(path)
        else:
            from ..playbook import Playbook

            Playbook.load(path)
    except Exception as exc:
        print(f"Invalid playbook {path}: {exc}", file=sys.stderr)
        return 1
    out = args.out or str(Path(path).parent / f"improved.{Path(path).name}")
    final_yaml, lines = _run_optimize(
        path,
        rounds=args.rounds,
        n=args.n,
        personas_path=args.personas,
        llm=args.llm,
        candidate_llm=args.candidate_llm,
        user_llm=args.user_llm,
    )
    Path(out).write_text(final_yaml, encoding="utf-8")
    for line in lines:
        print(line)
    print(f"Wrote {out}")
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="superdialog")
    sub = parser.add_subparsers(dest="cmd", required=True)

    chat = sub.add_parser("chat", help="Interactive REPL against a flow or playbook")
    chat.add_argument(
        "--flow",
        default="flow.json",
        help="Path to flow JSON (default: ./flow.json); a YAML/JSON file with "
        "a top-level 'journeys' key is auto-detected as a playbook",
    )
    chat.add_argument(
        "--playbook",
        default=None,
        help="Path to a playbook (YAML/JSON); forces the playbook REPL",
    )
    chat.add_argument(
        "--simple",
        default=None,
        help="Path to a simple-format playbook (YAML/JSON); compiles then runs",
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

    opt = sub.add_parser(
        "optimize", help="Reflectively improve a playbook's prose via self-play"
    )
    opt.add_argument(
        "--playbook",
        required=True,
        help="Path to a playbook (full or simple format)",
    )
    opt.add_argument("--rounds", type=int, default=3)
    opt.add_argument(
        "--n", type=int, default=1, help="Eval sessions per persona per side"
    )
    opt.add_argument(
        "--personas", default=None, help="Path to a PersonaSpec list (YAML/JSON)"
    )
    opt.add_argument("--llm", default="openai/gpt-4o-mini")
    opt.add_argument(
        "--candidate-llm",
        default=None,
        help="Override the reflecting LLM (default: --llm)",
    )
    opt.add_argument(
        "--user-llm",
        default=None,
        help="Override the caller-simulator LLM (default: --llm)",
    )
    opt.add_argument(
        "--out",
        default=None,
        help="Output path (default: improved.<name>, same format)",
    )
    opt.set_defaults(fn=_cmd_optimize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    rc = args.fn(args)
    return int(rc or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
