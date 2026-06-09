"""Eval: audit a real session traversal against the flow.

Run:
    pytest tests/evals/test_audit_session.py -s -v \
        --flow /path/to/flow.json \
        --traversal /path/to/traversal_*.json
"""
from __future__ import annotations

import pytest

from superdialog.machine.eval.session_auditor import SessionAuditor


@pytest.mark.anyio
async def test_audit_session(flow, traversal_path, llm_fn) -> None:
    auditor = SessionAuditor(flow=flow, llm_fn=llm_fn)
    report = await auditor.audit_file(traversal_path)
    print("\n" + report.to_markdown())

    assert report.overall_score >= 0.0
    if report.critical_issues:
        print("\nCRITICAL ISSUES:")
        for issue in report.critical_issues:
            print(f"  - {issue}")