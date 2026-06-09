# src/superdialog/machine/eval/corpus_generator.py
from __future__ import annotations

import json
import logging
import re
from collections import deque
from typing import Any, Callable

from superdialog.flow.models import ConversationFlow
from superdialog.machine.eval.models import EdgeTest, PathStep, PathTest, TestCorpus

logger = logging.getLogger(__name__)

LLMFn = Callable[[list[dict[str, Any]]], Any]


class CorpusGenerator:
    def __init__(
        self,
        flow: ConversationFlow,
        llm_fn: LLMFn,
        utterances_per_edge: int = 3,
        negative_per_edge: int = 2,
    ) -> None:
        self._flow = flow
        self._llm_fn = llm_fn
        self._utterances_per_edge = utterances_per_edge
        self._negative_per_edge = negative_per_edge
        self._node_map = {n.id: n for n in flow.nodes}

    # ── Public properties ────────────────────────────────────────────────

    @property
    def faq_pairs(self) -> list[tuple[str, str]]:
        return self._extract_faq(self._flow.system_prompt)

    @property
    def guardrails(self) -> list[str]:
        return self._extract_guardrails(self._flow.system_prompt)

    @property
    def has_multilingual(self) -> bool:
        return self._detect_language_switching(self._flow.system_prompt)

    # ── Public methods ────────────────────────────────────────────────────

    async def generate_edge_tests(self) -> list[EdgeTest]:
        tests: list[EdgeTest] = []
        for node in self._flow.nodes:
            for edge in node.edges:
                siblings = self._format_sibling_edges(node.id, exclude_edge_id=edge.id)
                lang_hint = self._language_instruction()
                prompt = (
                    f"You are generating test data for a voice dialog flow.\n"
                    f"Node instruction: \"{node.instruction or node.static_text or ''}\"\n"
                    f"This edge fires when: \"{edge.condition}\"\n"
                    f"Sibling edges:\n{siblings}\n"
                    f"Give {self._utterances_per_edge} utterances a real caller might say that SHOULD trigger "
                    f"this edge, and {self._negative_per_edge} utterances that should NOT trigger it "
                    f"(to test false-positive resistance).\n"
                    f"{lang_hint}"
                    f'Return JSON only: {{"utterances": [...], "negative_utterances": [...]}}'
                )
                messages = [{"role": "user", "content": prompt}]
                try:
                    raw = await self._llm_fn(messages)
                    data = self._parse_json(raw)
                    utterances = data.get("utterances", [])
                    negatives = data.get("negative_utterances", [])
                except Exception as exc:
                    logger.warning("CorpusGenerator LLM failed for edge %s: %s", edge.id, exc)
                    utterances = []
                    negatives = []
                tests.append(
                    EdgeTest(
                        node_id=node.id,
                        edge_id=edge.id,
                        condition=edge.condition or "",
                        utterances=utterances,
                        negative_utterances=negatives,
                    )
                )
        return tests

    async def generate_path_tests(self) -> list[PathTest]:
        paths = self._enumerate_paths()
        path_tests: list[PathTest] = []
        for i, path in enumerate(paths):
            steps: list[PathStep] = []
            for edge_id, from_node_id, to_node_id in path:
                node = self._node_map.get(from_node_id)
                edge = next(
                    (e for e in (node.edges if node else []) if e.id == edge_id),
                    None,
                )
                if node is None or edge is None:
                    steps.append(
                        PathStep(
                            utterance=f"[trigger:{edge_id}]",
                            expected_edge=edge_id,
                            expected_node=to_node_id,
                        )
                    )
                    continue
                prompt = (
                    f"Generate one short utterance a caller might say to trigger the edge "
                    f"\"{edge.condition}\" from node \"{node.instruction or ''}\". "
                    f"Return only the utterance text, no quotes."
                )
                messages = [{"role": "user", "content": prompt}]
                try:
                    raw = await self._llm_fn(messages)
                    data = self._parse_json(raw) if raw.strip().startswith("{") else None
                    utterance = (
                        data.get("utterances", [""])[0]
                        if data
                        else raw.strip().strip('"').strip("'")
                    )
                    if not utterance:
                        utterance = f"[trigger:{edge_id}]"
                except Exception:
                    utterance = f"[trigger:{edge_id}]"
                steps.append(
                    PathStep(
                        utterance=utterance,
                        expected_edge=edge_id,
                        expected_node=to_node_id,
                    )
                )
            path_tests.append(PathTest(name=f"path_{i + 1}", steps=steps))
        return path_tests

    async def generate_corpus(self, flow_file: str = "") -> TestCorpus:
        edge_tests = await self.generate_edge_tests()
        path_tests = await self.generate_path_tests()
        return TestCorpus(
            flow_file=flow_file,
            edge_tests=edge_tests,
            path_tests=path_tests,
            generated_by="corpus_generator",
            reviewed=False,
        )

    # ── Static helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_faq(system_prompt: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        lines = system_prompt.splitlines()
        q: str | None = None
        for line in lines:
            line = line.strip()
            if line.startswith("Q:"):
                q = line[2:].strip()
            elif line.startswith("A:") and q is not None:
                pairs.append((q, line[2:].strip()))
                q = None
        return pairs

    @staticmethod
    def _extract_guardrails(system_prompt: str) -> list[str]:
        guardrails: list[str] = []
        in_section = False
        for line in system_prompt.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("GUARDRAIL"):
                in_section = True
                continue
            if in_section:
                if stripped.startswith("-"):
                    guardrails.append(stripped[1:].strip())
                elif stripped and not stripped.startswith("-"):
                    in_section = False
        return guardrails

    @staticmethod
    def _detect_language_switching(system_prompt: str) -> bool:
        lower = system_prompt.lower()
        keywords = ["hindi", "hinglish", "multilingual", "bilingual", "language"]
        return any(k in lower for k in keywords)

    def _format_sibling_edges(
        self,
        node_id: str,
        exclude_edge_id: str | None = None,
    ) -> str:
        node = self._node_map.get(node_id)
        if node is None or not node.edges:
            return "(no edges)"
        lines: list[str] = []
        for edge in node.edges:
            marker = " [TARGET]" if edge.id == exclude_edge_id else ""
            lines.append(f'  - id: "{edge.id}" condition: "{edge.condition}"{marker}')
        return "\n".join(lines) if lines else "(no edges)"

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        fenced = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        return json.loads(text)

    def _language_instruction(self) -> str:
        if not self.has_multilingual:
            return ""
        return (
            "Include utterances in both English and Hindi/Hinglish since this flow supports "
            "multilingual callers.\n"
        )

    # ── Internal path enumeration ──────────────────────────────────────────

    def _enumerate_paths(self) -> list[list[tuple[str, str, str]]]:
        paths: list[list[tuple[str, str, str]]] = []
        queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque()
        queue.append((self._flow.initial_node, []))
        visited_states: set[frozenset] = set()
        MAX_PATHS = 20

        while queue and len(paths) < MAX_PATHS:
            node_id, current_path = queue.popleft()
            node = self._node_map.get(node_id)
            if node is None:
                continue
            if node.is_final or not node.edges:
                paths.append(current_path)
                continue
            state_key = frozenset((node_id, tuple(s[0] for s in current_path)))
            if state_key in visited_states:
                if current_path:
                    paths.append(current_path)
                continue
            visited_states.add(state_key)
            for edge in node.edges:
                if edge.target_node_id:
                    queue.append(
                        (edge.target_node_id, current_path + [(edge.id, node_id, edge.target_node_id)])
                    )
        return paths if paths else [[]]