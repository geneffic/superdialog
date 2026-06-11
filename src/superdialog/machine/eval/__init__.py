from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Union

from superdialog.machine.eval.models import AuditReport, EvalReport


async def run_eval(
    flow_path: str,
    traversal_path: str | None = None,
    *,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    utterances_per_edge: int = 2,
    negative_per_edge: int = 1,
) -> Union[AuditReport, EvalReport]:
    """Run eval against a flow.

    If traversal_path given: audit that session (SessionAuditor → AuditReport).
    Otherwise: generate corpus + run FlowEvaluator → EvalReport.

    Args:
        flow_path: Path to flow JSON.
        traversal_path: Optional path to traversal JSON from a real session.
        model: OpenAI model to use as eval LLM.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        utterances_per_edge: Corpus utterances per edge (no-traversal mode).
        negative_per_edge: Negative utterances per edge (no-traversal mode).
    """
    from superdialog.flow import load_flow

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required: pass api_key= or set OPENAI_API_KEY")

    import openai

    client = openai.AsyncOpenAI(api_key=key)

    async def llm_fn(messages: list[dict[str, Any]]) -> str:
        resp = await client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content

    flow = load_flow(flow_path)

    if traversal_path is not None:
        from superdialog.machine.eval.session_auditor import SessionAuditor

        auditor = SessionAuditor(flow=flow, llm_fn=llm_fn)
        return await auditor.audit_file(traversal_path)

    from superdialog.machine.eval.corpus_generator import CorpusGenerator
    from superdialog.machine.eval.evaluator import FlowEvaluator

    generator = CorpusGenerator(
        flow=flow,
        llm_fn=llm_fn,
        utterances_per_edge=utterances_per_edge,
        negative_per_edge=negative_per_edge,
    )
    corpus = await generator.generate_corpus(flow_file=flow_path)

    evaluator = FlowEvaluator(
        flow=flow,
        llm_factory=lambda _model_id: llm_fn,
    )
    return await evaluator.eval_corpus(corpus, model_ids=[model])


__all__ = ["run_eval", "AuditReport", "EvalReport"]