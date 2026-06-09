"""Eval: generate a test corpus from a flow using LLM.

Run:
    pytest tests/evals/test_generate_corpus.py -s -v \
        --flow /path/to/flow.json

Saves corpus to <flow_name>_corpus.json next to the flow file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from superdialog.machine.eval.corpus_generator import CorpusGenerator


@pytest.mark.anyio
async def test_generate_corpus(flow, flow_path, llm_fn) -> None:
    generator = CorpusGenerator(
        flow=flow,
        llm_fn=llm_fn,
        utterances_per_edge=3,
        negative_per_edge=2,
    )
    corpus = await generator.generate_corpus(flow_file=flow_path)

    print(f"\nGenerated {len(corpus.edge_tests)} edge tests")
    print(f"Generated {len(corpus.path_tests)} path tests")
    for et in corpus.edge_tests:
        print(f"\n  edge: {et.edge_id} (node: {et.node_id})")
        print(f"    + {et.utterances}")
        print(f"    - {et.negative_utterances}")

    out_path = Path(flow_path).with_suffix("_corpus.json")
    out_path.write_text(
        json.dumps(corpus.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nCorpus saved → {out_path}")

    assert len(corpus.edge_tests) > 0