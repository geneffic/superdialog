from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from superdialog.flow.models import ConversationFlow
from superdialog.machine.eval.models import (
    AuditReport,
    EdgeVerdict,
    PathViolation,
    ResponseVerdict,
)

logger = logging.getLogger(__name__)

LLMFn = Callable[[list[dict[str, Any]]], Any]


class SessionAuditor:
    def __init__(
        self,
        flow: ConversationFlow,
        llm_fn: LLMFn | None = None,
    ) -> None:
        self._flow = flow
        self._llm_fn = llm_fn
        self._node_map = {n.id: n for n in flow.nodes}
        self._edge_map: dict[str, tuple[str, Any]] = {}
        for node in flow.nodes:
            for edge in node.edges:
                self._edge_map[edge.id] = (node.id, edge)
        for edge in getattr(flow, "global_edges", []):
            self._edge_map[edge.id] = ("__global__", edge)

    async def audit_file(self, path: str | Path) -> AuditReport:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return await self.audit(data)

    async def audit(self, traversal: dict[str, Any]) -> AuditReport:
        steps: list[dict[str, Any]] = traversal.get("traversal", [])
        report = AuditReport(
            session_id=traversal.get("session_id", ""),
            flow_file=traversal.get("flow_file", ""),
            final_node=steps[-1]["to_node"] if steps else "",
            reached_final=traversal.get("is_complete", False),
        )

        self._run_layer1(steps, report)
        await self._run_layer2(steps, report)
        await self._run_layer3(steps, report)
        self._run_layer4(steps, report)
        self._compute_overall(report)
        return report

    def _run_layer1(
        self,
        steps: list[dict[str, Any]],
        report: AuditReport,
    ) -> None:
        violations: list[PathViolation] = []
        for step in steps:
            step_num = step.get("step", 0)
            edge_id = step.get("edge_id")
            from_node = step.get("from_node")
            to_node = step.get("to_node")

            if edge_id is None:
                continue

            if edge_id not in self._edge_map:
                violations.append(
                    PathViolation(
                        step=step_num,
                        edge_id=edge_id,
                        from_node=from_node,
                        to_node=to_node,
                        reason=f"edge '{edge_id}' not found in flow",
                    )
                )
                continue

            edge_from_node, edge_obj = self._edge_map[edge_id]
            if edge_from_node != "__global__" and edge_obj.target_node_id != to_node:
                violations.append(
                    PathViolation(
                        step=step_num,
                        edge_id=edge_id,
                        from_node=from_node,
                        to_node=to_node,
                        reason=(
                            f"edge '{edge_id}' targets '{edge_obj.target_node_id}' "
                            f"but session went to '{to_node}'"
                        ),
                    )
                )

        report.path_violations = violations
        report.path_valid = len(violations) == 0

    async def _run_layer2(
        self,
        steps: list[dict[str, Any]],
        report: AuditReport,
    ) -> None:
        if self._llm_fn is None:
            return
        verdicts: list[EdgeVerdict] = []
        for step in steps:
            edge_id = step.get("edge_id")
            from_node = step.get("from_node")
            if edge_id is None or from_node is None:
                continue

            node = self._node_map.get(from_node)
            if node is None:
                continue

            user_msg = step.get("user_message") or ""
            bot_msg = step.get("bot_message") or ""
            instruction = step.get("node_instruction") or ""
            sibling_edges = "\n".join(
                f'  - id: "{e.id}" condition: "{e.condition}"'
                for e in node.edges
            )

            prompt = (
                f"You are auditing a voice agent conversation.\n"
                f"Node instruction: \"{instruction}\"\n"
                f"Available edges:\n{sibling_edges}\n"
                f"Agent said: \"{bot_msg}\"\n"
                f"Caller replied: \"{user_msg}\"\n"
                f"Agent then took edge: \"{edge_id}\"\n\n"
                f"Was this the correct edge? If not, which edge should have been taken?\n"
                f'Return JSON: {{"correct": true/false, "confidence": "high"/"medium"/"low", '
                f'"preferred_edge": null or "edge_id", "reason": "..."}}'
            )
            try:
                raw = await self._llm_fn([{"role": "user", "content": prompt}])
                raw = raw.strip()
                if raw.startswith("```"):
                    import re
                    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", raw, re.DOTALL)
                    raw = m.group(1).strip() if m else raw
                data = json.loads(raw)
                verdict = EdgeVerdict(
                    step=step.get("step", 0),
                    edge_id=edge_id,
                    from_node=from_node,
                    correct=bool(data.get("correct", True)),
                    confidence=data.get("confidence", "high"),
                    preferred_edge=data.get("preferred_edge"),
                    reason=data.get("reason", ""),
                )
            except Exception as exc:
                logger.warning("Layer2 LLM failed at step %s: %s", step.get("step"), exc)
                verdict = EdgeVerdict(
                    step=step.get("step", 0),
                    edge_id=edge_id,
                    from_node=from_node,
                    correct=True,
                    confidence="low",
                    reason=f"eval error: {exc}",
                )
            verdicts.append(verdict)

        report.edge_verdicts = verdicts
        if verdicts:
            correct = sum(1 for v in verdicts if v.correct)
            report.edge_accuracy = correct / len(verdicts)

    async def _run_layer3(
        self,
        steps: list[dict[str, Any]],
        report: AuditReport,
    ) -> None:
        if self._llm_fn is None:
            return
        verdicts: list[ResponseVerdict] = []
        leaks: list[str] = []
        for step in steps:
            bot_msg = step.get("bot_message")
            if not bot_msg:
                continue
            instruction = step.get("node_instruction") or ""

            prompt = (
                f"You are auditing a voice agent response in a dialog flow system.\n"
                f"Node instruction given to the agent: \"{instruction}\"\n"
                f"Agent said: \"{bot_msg}\"\n\n"
                f"Evaluate on a scale of 1-5:\n"
                f"1 = Wrong (ignores instruction, or contains routing/flow-logic language)\n"
                f"3 = Acceptable (follows instruction, minor issues)\n"
                f"5 = Perfect (concise, correct, natural spoken language)\n\n"
                f"routing_leak means the agent accidentally spoke internal flow logic aloud — "
                f"e.g. mentioning edge names, node names, conditions, transition instructions, "
                f"or any text that reads like backend instructions rather than natural speech.\n\n"
                f'Return JSON only: {{"score": 1-5, "routing_leak": true/false, "issues": [...]}}'
            )
            try:
                raw = await self._llm_fn([{"role": "user", "content": prompt}])
                raw = raw.strip()
                if raw.startswith("```"):
                    import re
                    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", raw, re.DOTALL)
                    raw = m.group(1).strip() if m else raw
                data = json.loads(raw)
                is_leak = bool(data.get("routing_leak", False))
                if is_leak:
                    leaks.append(bot_msg)
                verdict = ResponseVerdict(
                    step=step.get("step", 0),
                    score=int(data.get("score", 3)),
                    issues=data.get("issues", []),
                    routing_leak=is_leak,
                    bot_message=bot_msg,
                )
            except Exception as exc:
                logger.warning("Layer3 LLM failed at step %s: %s", step.get("step"), exc)
                verdict = ResponseVerdict(
                    step=step.get("step", 0),
                    score=3,
                    bot_message=bot_msg,
                )
            verdicts.append(verdict)

        report.response_verdicts = verdicts
        report.routing_leaks = leaks
        if verdicts:
            report.response_quality = sum(v.score for v in verdicts) / len(verdicts)

    def _run_layer4(
        self,
        steps: list[dict[str, Any]],
        report: AuditReport,
    ) -> None:
        slot_coverage: dict[str, bool] = {}
        for step in steps:
            criteria = step.get("criteria")
            if not criteria:
                continue
            criteria_map: dict[str, bool] = criteria.get("criteria_map", {})
            for slot, met in criteria_map.items():
                if slot not in slot_coverage:
                    slot_coverage[slot] = met
                else:
                    slot_coverage[slot] = slot_coverage[slot] or met

        report.slot_coverage = slot_coverage
        if slot_coverage:
            captured = sum(1 for v in slot_coverage.values() if v)
            report.slot_completeness = captured / len(slot_coverage)
        else:
            report.slot_completeness = 1.0

    def _compute_overall(self, report: AuditReport) -> None:
        path_score = 1.0 if report.path_valid else 0.0
        edge_score = report.edge_accuracy if report.edge_verdicts else 1.0
        resp_score = (report.response_quality - 1) / 4 if report.response_verdicts else 1.0
        slot_score = report.slot_completeness

        report.overall_score = (
            path_score * 0.2
            + edge_score * 0.4
            + resp_score * 0.3
            + slot_score * 0.1
        )

        issues: list[str] = []
        for v in report.path_violations:
            issues.append(f"Path: {v.reason}")
        for v in report.edge_verdicts:
            if not v.correct:
                issues.append(
                    f"Edge step {v.step}: took '{v.edge_id}' — preferred '{v.preferred_edge}'"
                )
        for leak in report.routing_leaks:
            issues.append(f"Routing leak: {leak[:60]}")
        for slot, ok in report.slot_coverage.items():
            if not ok:
                issues.append(f"Slot not captured: {slot}")
        report.critical_issues = issues