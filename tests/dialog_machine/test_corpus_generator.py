"""Tests for CorpusGenerator -- script-aware LLM-powered test corpus generation.

Covers FAQ extraction, guardrail extraction, multilingual detection,
sibling edge formatting, and async generation with a mock LLM.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

# Stub livekit modules before any project imports
for _mod in [
    "livekit.agents",
    "livekit.agents.llm",
    "livekit.agents.llm.tool_context",
    "livekit.agents.voice",
    "livekit.api",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402

from superdialog.flow.models import ConversationFlow, Edge, FlowNode  # noqa: E402
from superdialog.machine.eval.corpus_generator import CorpusGenerator  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Kairali receptionist.
KNOWLEDGE BASE:
Q: What services do you offer?
A: We offer Ayurvedic treatments and consultations.
Q: What are your timings?
A: We are open 9am to 6pm daily.
GUARDRAILS:
- Never discuss pricing without manager approval
- Never share patient information
LANGUAGE: Respond in Hindi or English based on caller preference."""

SYSTEM_PROMPT_NO_MULTILINGUAL = """\
You are a simple assistant.
GUARDRAILS:
- Never reveal internal policies"""

SYSTEM_PROMPT_EMPTY = "You are a helpful bot."


def _make_flow(
    system_prompt: str = SYSTEM_PROMPT,
) -> ConversationFlow:
    """Build a simple 3-node flow for testing."""
    return ConversationFlow(
        system_prompt=system_prompt,
        initial_node="greeting",
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                instruction="Greet the caller and ask how to help.",
                edges=[
                    Edge(
                        id="e_book",
                        condition="User wants to book an appointment",
                        target_node_id="booking",
                    ),
                    Edge(
                        id="e_faq",
                        condition="User asks a general question",
                        target_node_id="faq_answer",
                    ),
                    Edge(
                        id="e_fallback",
                        condition="Unrecognised input",
                        target_node_id="greeting",
                        is_fallback=True,
                    ),
                ],
            ),
            FlowNode(
                id="booking",
                name="Booking",
                instruction="Collect date and time for appointment.",
                edges=[
                    Edge(
                        id="e_confirm",
                        condition="Appointment details confirmed",
                        target_node_id="farewell",
                    ),
                ],
            ),
            FlowNode(
                id="faq_answer",
                name="FAQ Answer",
                instruction="Answer the question from knowledge base.",
                edges=[
                    Edge(
                        id="e_done",
                        condition="User has no more questions",
                        target_node_id="farewell",
                    ),
                ],
            ),
            FlowNode(
                id="farewell",
                name="Farewell",
                static_text="Thank you for calling. Goodbye!",
                is_final=True,
                edges=[],
            ),
        ],
    )


@pytest.fixture
def flow() -> ConversationFlow:
    return _make_flow()


_MOCK_LLM_RESPONSE = json.dumps(
    {
        "utterances": ["test1", "test2", "test3"],
        "negative_utterances": ["neg1", "neg2"],
    }
)


async def _mock_llm(messages: list[dict[str, str]]) -> str:
    """Mock LLM that returns a valid JSON response."""
    return _MOCK_LLM_RESPONSE


@pytest.fixture
def generator(flow: ConversationFlow) -> CorpusGenerator:
    return CorpusGenerator(
        flow=flow,
        llm_fn=_mock_llm,
        utterances_per_edge=3,
        negative_per_edge=2,
    )


# ---------------------------------------------------------------------------
# Static method tests: FAQ extraction
# ---------------------------------------------------------------------------


class TestExtractFaq:
    """Test _extract_faq static method."""

    def test_extracts_pairs(self) -> None:
        pairs = CorpusGenerator._extract_faq(SYSTEM_PROMPT)
        assert len(pairs) == 2
        assert pairs[0][0] == "What services do you offer?"
        assert "Ayurvedic" in pairs[0][1]
        assert pairs[1][0] == "What are your timings?"
        assert "9am" in pairs[1][1]

    def test_empty_prompt_returns_no_pairs(self) -> None:
        pairs = CorpusGenerator._extract_faq(SYSTEM_PROMPT_EMPTY)
        assert pairs == []

    def test_faq_property(self, generator: CorpusGenerator) -> None:
        assert len(generator.faq_pairs) == 2
        assert generator.faq_pairs[0][0] == "What services do you offer?"


# ---------------------------------------------------------------------------
# Static method tests: Guardrail extraction
# ---------------------------------------------------------------------------


class TestExtractGuardrails:
    """Test _extract_guardrails static method."""

    def test_extracts_guardrails(self) -> None:
        guardrails = CorpusGenerator._extract_guardrails(SYSTEM_PROMPT)
        assert len(guardrails) == 2
        assert "pricing" in guardrails[0].lower()
        assert "patient" in guardrails[1].lower()

    def test_empty_prompt_returns_no_guardrails(self) -> None:
        guardrails = CorpusGenerator._extract_guardrails(SYSTEM_PROMPT_EMPTY)
        assert guardrails == []

    def test_single_guardrail(self) -> None:
        guardrails = CorpusGenerator._extract_guardrails(SYSTEM_PROMPT_NO_MULTILINGUAL)
        assert len(guardrails) == 1
        assert "internal policies" in guardrails[0]

    def test_guardrails_property(self, generator: CorpusGenerator) -> None:
        assert len(generator.guardrails) == 2


# ---------------------------------------------------------------------------
# Static method tests: Multilingual detection
# ---------------------------------------------------------------------------


class TestDetectLanguageSwitching:
    """Test _detect_language_switching static method."""

    def test_detects_hindi(self) -> None:
        assert CorpusGenerator._detect_language_switching(SYSTEM_PROMPT) is True

    def test_detects_multilingual_keyword(self) -> None:
        assert (
            CorpusGenerator._detect_language_switching("Support multilingual callers.")
            is True
        )

    def test_detects_language_keyword(self) -> None:
        assert (
            CorpusGenerator._detect_language_switching(
                "Respond in the caller's preferred language."
            )
            is True
        )

    def test_no_multilingual(self) -> None:
        assert CorpusGenerator._detect_language_switching(SYSTEM_PROMPT_EMPTY) is False

    def test_no_multilingual_guardrails_only(self) -> None:
        assert (
            CorpusGenerator._detect_language_switching(SYSTEM_PROMPT_NO_MULTILINGUAL)
            is False
        )

    def test_has_multilingual_property(self, generator: CorpusGenerator) -> None:
        assert generator.has_multilingual is True


# ---------------------------------------------------------------------------
# Sibling edge formatting
# ---------------------------------------------------------------------------


class TestFormatSiblingEdges:
    """Test _format_sibling_edges method."""

    def test_formats_all_edges(self, generator: CorpusGenerator) -> None:
        result = generator._format_sibling_edges("greeting")
        assert "e_book" in result
        assert "e_faq" in result
        assert "e_fallback" in result
        # No target marker when exclude_edge_id is empty
        assert "TARGET" not in result

    def test_marks_target_edge(self, generator: CorpusGenerator) -> None:
        result = generator._format_sibling_edges("greeting", exclude_edge_id="e_book")
        assert "e_book" in result
        assert "TARGET" in result
        # Only one TARGET marker
        assert result.count("TARGET") == 1

    def test_unknown_node_returns_no_edges(self, generator: CorpusGenerator) -> None:
        result = generator._format_sibling_edges("nonexistent")
        assert result == "(no edges)"

    def test_final_node_no_edges(self, generator: CorpusGenerator) -> None:
        result = generator._format_sibling_edges("farewell")
        assert result == "(no edges)"


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestParseJson:
    """Test _parse_json static method."""

    def test_plain_json(self) -> None:
        data = CorpusGenerator._parse_json('{"utterances": ["a"]}')
        assert data == {"utterances": ["a"]}

    def test_fenced_json(self) -> None:
        text = '```json\n{"utterances": ["a"]}\n```'
        data = CorpusGenerator._parse_json(text)
        assert data == {"utterances": ["a"]}

    def test_fenced_no_lang(self) -> None:
        text = '```\n{"utterances": ["a"]}\n```'
        data = CorpusGenerator._parse_json(text)
        assert data == {"utterances": ["a"]}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            CorpusGenerator._parse_json("not json at all")


# ---------------------------------------------------------------------------
# Async tests: edge test generation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestGenerateEdgeTests:
    """Test generate_edge_tests with mock LLM."""

    async def test_generates_edge_tests_for_all_edges(
        self, generator: CorpusGenerator
    ) -> None:
        edge_tests = await generator.generate_edge_tests()
        # Flow has 5 edges total: 3 on greeting + 1 on booking + 1 on faq
        assert len(edge_tests) == 5

    async def test_edge_test_has_utterances(self, generator: CorpusGenerator) -> None:
        edge_tests = await generator.generate_edge_tests()
        first = edge_tests[0]
        assert first.node_id == "greeting"
        assert first.edge_id == "e_book"
        assert first.utterances == ["test1", "test2", "test3"]
        assert first.negative_utterances == ["neg1", "neg2"]

    async def test_edge_test_condition_populated(
        self, generator: CorpusGenerator
    ) -> None:
        edge_tests = await generator.generate_edge_tests()
        first = edge_tests[0]
        assert "appointment" in first.condition.lower()

    async def test_llm_failure_produces_empty_utterances(
        self, flow: ConversationFlow
    ) -> None:
        async def _failing_llm(
            messages: list[dict[str, str]],
        ) -> str:
            raise RuntimeError("LLM unavailable")

        gen = CorpusGenerator(
            flow=flow,
            llm_fn=_failing_llm,
            utterances_per_edge=3,
            negative_per_edge=2,
        )
        edge_tests = await gen.generate_edge_tests()
        assert len(edge_tests) == 5
        for et in edge_tests:
            assert et.utterances == []
            assert et.negative_utterances == []


# ---------------------------------------------------------------------------
# Async tests: path test generation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestGeneratePathTests:
    """Test generate_path_tests with mock LLM."""

    async def test_generates_path_tests(self, generator: CorpusGenerator) -> None:
        path_tests = await generator.generate_path_tests()
        # Should have at least 1 path (greeting -> booking -> farewell)
        assert len(path_tests) >= 1

    async def test_path_test_names_sequential(self, generator: CorpusGenerator) -> None:
        path_tests = await generator.generate_path_tests()
        for i, pt in enumerate(path_tests):
            assert pt.name == f"path_{i + 1}"

    async def test_path_steps_have_utterances(self, generator: CorpusGenerator) -> None:
        path_tests = await generator.generate_path_tests()
        for pt in path_tests:
            assert len(pt.steps) >= 1
            for step in pt.steps:
                assert step.utterance == "test1"

    async def test_path_steps_have_expected_fields(
        self, generator: CorpusGenerator
    ) -> None:
        path_tests = await generator.generate_path_tests()
        for pt in path_tests:
            for step in pt.steps:
                assert step.expected_edge
                assert step.expected_node

    async def test_llm_failure_produces_fallback_utterance(
        self, flow: ConversationFlow
    ) -> None:
        async def _failing_llm(
            messages: list[dict[str, str]],
        ) -> str:
            raise RuntimeError("LLM unavailable")

        gen = CorpusGenerator(
            flow=flow,
            llm_fn=_failing_llm,
        )
        path_tests = await gen.generate_path_tests()
        for pt in path_tests:
            for step in pt.steps:
                assert step.utterance.startswith("[trigger:")


# ---------------------------------------------------------------------------
# Async tests: full corpus generation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestGenerateCorpus:
    """Test generate_corpus end-to-end with mock LLM."""

    async def test_corpus_structure(self, generator: CorpusGenerator) -> None:
        corpus = await generator.generate_corpus(flow_file="test_flow.json")
        assert corpus.flow_file == "test_flow.json"
        assert corpus.generated_by == "corpus_generator"
        assert corpus.reviewed is False

    async def test_corpus_has_edge_tests(self, generator: CorpusGenerator) -> None:
        corpus = await generator.generate_corpus()
        assert len(corpus.edge_tests) == 5

    async def test_corpus_has_path_tests(self, generator: CorpusGenerator) -> None:
        corpus = await generator.generate_corpus()
        assert len(corpus.path_tests) >= 1

    async def test_corpus_default_flow_file(self, generator: CorpusGenerator) -> None:
        corpus = await generator.generate_corpus()
        assert corpus.flow_file == ""


# ---------------------------------------------------------------------------
# Language instruction
# ---------------------------------------------------------------------------


class TestLanguageInstruction:
    """Test _language_instruction method."""

    def test_multilingual_instruction(self, generator: CorpusGenerator) -> None:
        instruction = generator._language_instruction()
        assert "Hindi" in instruction
        assert "Hinglish" in instruction

    def test_no_multilingual_instruction(self) -> None:
        flow = _make_flow(system_prompt=SYSTEM_PROMPT_EMPTY)
        gen = CorpusGenerator(flow=flow, llm_fn=_mock_llm)
        assert gen._language_instruction() == ""


# ---------------------------------------------------------------------------
# Edge cases: flow with no edges
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestEdgeCases:
    """Edge cases for corpus generation."""

    async def test_flow_with_single_final_node(self) -> None:
        flow = ConversationFlow(
            system_prompt="Simple bot.",
            initial_node="only",
            nodes=[
                FlowNode(
                    id="only",
                    name="Only Node",
                    instruction="Just say hi.",
                    is_final=True,
                    edges=[],
                ),
            ],
        )
        gen = CorpusGenerator(flow=flow, llm_fn=_mock_llm)
        edge_tests = await gen.generate_edge_tests()
        assert edge_tests == []

        path_tests = await gen.generate_path_tests()
        # Analyzer may produce a path with zero steps for a final node
        for pt in path_tests:
            assert pt.steps == []
