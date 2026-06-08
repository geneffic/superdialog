"""InstructionComposer — shared instruction building for all flow agents.

Consolidates voice preamble, identity extraction, language filtering,
template rendering, and language utterance directives into a single
reusable module.  Used by SimpleFlowAgent and FlowMachineLiteV2Agent.

Usage::

    from superdialog.machine.composer import InstructionComposer

    composer = InstructionComposer(machine=machine, lang="en", tone="professional")
    instructions = composer.compose(enriched_instructions, system_prompt)
"""

from __future__ import annotations

import logging
import re
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---- Regex patterns ----
_LANG_LINE_RE = re.compile(r"^\[([A-Z]{2})\]\s*", re.MULTILINE)
_TEMPLATE_VAR_RE = re.compile(r"\{\{(.+?)\}\}")
# Block-marker regex used by ``select_language_content``: matches a
# ``[XX]`` tag and everything up to (but not including) the next tag or
# end-of-string. Spans across newlines (re.DOTALL).
_LANG_MARKER_RE = re.compile(r"\[([A-Z]{2})\]\s*(.+?)(?=\n?\[[A-Z]{2}\]|\Z)", re.DOTALL)

_LANG_TAG_MAP = {
    "en": "EN",
    "hi": "HI",
    "english": "EN",
    "hindi": "HI",
}

# ---- Language utterance directive ----
LANGUAGE_UTTERANCE_DIRECTIVE = (
    "LANGUAGE RULE: Speak in ONE language only per response. "
    "If the text contains [EN]/[HI] markers, use ONLY the "
    "active language version. Never say the same thing in "
    "two languages."
)

# ---- Timing directive ----
TIMING_DIRECTIVE = (
    "TIMING RULE: When entering a new step, you MUST speak your "
    "message FIRST. Do NOT call any transition tool until the "
    "caller has responded. Never skip a step even if you think "
    "the caller already answered — each step must be spoken aloud.\n"
    "EXCEPTION: If the step instruction explicitly says 'proceed "
    "immediately' or 'no caller response needed', call the "
    "transition tool right after speaking — do NOT wait."
)

# ---- Flow script adherence directive ----
FLOW_SCRIPT_DIRECTIVE = (
    "SCRIPT ADHERENCE (STRICT): You are following a structured "
    "conversation flow. Each step has a script you MUST follow.\n"
    "- Say EXACTLY what the script says. Small natural paraphrases "
    "are allowed but do NOT add information, merge steps, or skip ahead.\n"
    "- Do NOT volunteer extra questions, confirmations, or filler "
    "beyond what the script specifies.\n"
    "- Do NOT repeat the disclaimer, security warnings, or any "
    "prior step content unless the script explicitly says to.\n"
    "- After speaking, WAIT for the caller to respond before "
    "calling any transition tool.\n"
    "- If the script says 'proceed immediately' or 'no caller "
    "response needed', call the transition tool right after speaking."
)

# ---- Adaptive identity thresholds ----
_IDENTITY_FULL_THRESHOLD = 2000
_IDENTITY_TRUNCATE_LENGTH = 2000


# ------------------------------------------------------------------
# Text processing utilities
# ------------------------------------------------------------------


def select_language_content(
    text: str,
    language: str,
    fallback: Literal["warn", "english", "raise"] = "warn",
) -> str:
    """Single source of truth for language-marker filtering.

    Replaces the 5 historically overlapping functions (filter_language_markers,
    extract_speech_text, filter_language, extract_speech_only, process_text)
    with one explicit policy. Closes GAP-4 — mid-conversation language switches
    caused by inconsistent fallback semantics across the old functions.

    Behaviour:
        * No markers at all in ``text``: returns ``text`` unchanged (pass-through).
        * Marker for the requested ``language`` exists: returns its content.
        * Marker missing for ``language``:
            - fallback="warn":    emit UserWarning, return English content if
                                  present else the original text.
            - fallback="english": silently return English content if present
                                  else the original text.
            - fallback="raise":   raise ValueError.

    Language codes are lowercased for comparison. Block markers in the form
    ``[EN] ... [HI] ... [ES] ...`` are recognised — content for a marker
    extends until the next ``[XX]`` marker or end-of-string.
    """
    if not text:
        return text

    matches = list(_LANG_MARKER_RE.finditer(text))
    if not matches:
        return text  # unmarked passthrough

    by_lang: dict[str, str] = {}
    for m in matches:
        key = m.group(1).lower()
        if key not in by_lang:  # first match wins — preserves multi-step node ordering
            by_lang[key] = m.group(2).strip()
    lang = language.lower().strip() if language else "en"
    # Honour the existing alias map (english -> en, hindi -> hi, ...).
    alias = _LANG_TAG_MAP.get(lang)
    if alias is not None:
        lang = alias.lower()

    if lang in by_lang:
        return by_lang[lang]

    if fallback == "raise":
        raise ValueError(
            f"composer: text has no marker for language {lang!r}; "
            f"available={sorted(by_lang)}"
        )

    if fallback == "warn":
        warnings.warn(
            f"composer: missing marker for language {lang!r}; "
            f"falling back to English (available={sorted(by_lang)})",
            UserWarning,
            stacklevel=2,
        )

    # "warn" and "english" both reach here
    return by_lang.get("en", text)


def filter_language_markers(text: str, lang: str) -> str:
    """DEPRECATED — use ``select_language_content`` directly.

    Thin shim that delegates to ``select_language_content`` with the
    default ``fallback="warn"`` policy. Original line-based filtering is
    superseded by the block-marker semantics in the canonical function;
    this preserves the public signature for internal callers.
    """
    return select_language_content(text, lang, fallback="warn")


def extract_speech_text(text: str, machine: Any, lang: str) -> str | None:
    """DEPRECATED — use ``select_language_content`` directly.

    Returns None when ``text`` is empty or carries no language markers,
    matching the original contract (the caller falls back to
    ``generate_reply``). Otherwise delegates to ``select_language_content``
    (``fallback="warn"``) and runs the result through ``render_template``.

    Post-processing: strips everything from ``<wait for response>`` onwards
    so that routing/capture instructions embedded in the [HI]/[EN] block
    are never passed to TTS. This matches the Kairali-style flow format
    where routing metadata follows the speech text inside the same block.
    """
    if not text:
        return None
    if not _LANG_MARKER_RE.search(text):
        return None
    speech = select_language_content(text, lang, fallback="warn")
    # Strip routing/metadata section that follows speech text.
    # <wait for response> is the canonical boundary between speech and instructions.
    _STRIP_MARKERS = ("<wait for response>", "\nROUTING:", "\nCapture only")
    for marker in _STRIP_MARKERS:
        idx = speech.find(marker)
        if idx >= 0:
            speech = speech[:idx].strip()
    return render_template(speech, machine) if speech else None


def normalize_template_vars(text: str) -> str:
    """Replace '/' with '_' inside {{ }} blocks for Jinja2 compat."""
    return _TEMPLATE_VAR_RE.sub(
        lambda m: "{{" + m.group(1).replace("/", "_") + "}}", text
    )


def get_time_context() -> dict[str, str]:
    """Build timezone-aware time variables for template rendering."""
    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist)
    now_utc = datetime.now(timezone.utc)
    return {
        "current_time_Asia_Kolkata": now_ist.strftime("%I:%M %p IST"),
        "current_time": now_utc.strftime("%I:%M %p UTC"),
        "current_date": now_utc.strftime("%A, %B %d, %Y"),
    }


def render_template(text: str, machine: Any) -> str:
    """Render Jinja2 templates with machine userdata and environment.

    Uses ChainableUndefined so unknown variables render as empty strings.
    """
    if "{{" not in text:
        return text
    try:
        from jinja2 import BaseLoader, ChainableUndefined, Environment

        text = normalize_template_vars(text)

        # Autoescape intentionally disabled: rendered text is an LLM
        # instruction prompt (never HTML), so XSS is not applicable.
        env = Environment(  # nosec B701
            loader=BaseLoader(),
            undefined=ChainableUndefined,
            autoescape=False,
        )
        variables = getattr(getattr(machine, "context", None), "data", None)
        variables = getattr(variables, "variables", None) or {}
        flow_env = getattr(machine._flow, "environment_variables", {}) or {}
        # node_slots still needed for backward compat
        node_slots = getattr(getattr(machine, "context", None), "node_slots", {}) or {}
        all_slots: dict[str, Any] = {}
        for slot_dict in node_slots.values():
            all_slots.update(slot_dict)

        context: dict[str, Any] = {
            "env": flow_env,
            "userdata": dict(variables),
            "actions": dict(variables),
        }
        context.update(all_slots)
        context.update(variables)
        context.update(get_time_context())

        template = env.from_string(text)
        return template.render(**context)
    except Exception as exc:
        logger.warning("[Composer] template render failed: %s", exc)
        return text


def extract_identity(system_prompt: str) -> str:
    """Extract agent identity from flow system_prompt (~4K adaptive).

    Short prompts (<=4K chars) are included fully.
    Long prompts (>4K chars) are truncated at ~4K chars at a line boundary.
    """
    if not system_prompt:
        return ""
    if len(system_prompt) <= _IDENTITY_FULL_THRESHOLD:
        return system_prompt
    return system_prompt[:_IDENTITY_TRUNCATE_LENGTH].rsplit("\n", 1)[0]


def build_voice_preamble(
    tone: str,
    identity: str,
    *,
    include_leadership: bool = True,
) -> str:
    """Assemble static voice rules preamble.

    Order: VOICE_CORE > LEADERSHIP_RULES (or FLOW_SCRIPT) > TONE > identity.
    """
    from superdialog.machine._prompts import (
        LEADERSHIP_RULES,
        TONE_CASUAL,
        TONE_PROFESSIONAL,
        VOICE_CORE,
    )

    tone_block = TONE_PROFESSIONAL
    if tone and tone.strip().lower() == "casual":
        tone_block = TONE_CASUAL

    sections = [VOICE_CORE.strip()]
    if include_leadership:
        sections.append(LEADERSHIP_RULES.strip())
    else:
        sections.append(FLOW_SCRIPT_DIRECTIVE)
    sections.append(tone_block.strip())
    if identity:
        sections.append(identity.strip())
    return "\n\n".join(sections)


def resolve_language(machine: Any) -> str:
    """Determine active language from machine context."""
    ctx = getattr(machine, "context", None)
    # Try context.agent_language (works for both real FlowContext via shim and mocks)
    lang = getattr(ctx, "agent_language", "") or ""
    if isinstance(lang, str) and lang.strip():
        return lang.lower().strip()
    flow_lang = getattr(getattr(machine, "_flow", None), "agent_language", "") or ""
    if isinstance(flow_lang, str) and flow_lang.strip():
        return flow_lang.lower().strip()
    return "en"


def resolve_active_language(state: Any, machine: Any) -> str:
    """Determine active language: machine > flow > state config > 'en'.

    When the config fallback fires, seeds ``machine.context`` via
    ``set_language()`` so the config is never consulted again.
    """
    lang = resolve_language(machine)
    if lang != "en":
        return lang

    cfg = getattr(state, "config", {}) or {}
    for key in ("language", "preferred_language"):
        val = cfg.get(key, "")
        if val:
            lang = val.lower().strip()
            # Seed machine context so config is never needed again
            if hasattr(machine, "set_language"):
                machine.set_language(lang)
            return lang

    return lang


def filter_language(text: str, target_lang: str) -> str:
    """DEPRECATED — use ``select_language_content`` directly.

    Thin shim that delegates to ``select_language_content`` with the
    default ``fallback="warn"`` policy. The original line-based
    implementation has been consolidated; non-tagged lines are no longer
    preserved separately (the canonical function uses block-marker
    semantics, which is the consistent behaviour across all five
    historical filters).
    """
    return select_language_content(text, target_lang, fallback="warn")


def extract_speech_only(text: str, target_lang: str) -> str | None:
    """DEPRECATED — use ``select_language_content`` directly.

    Returns None when ``text`` is empty or carries no language markers,
    so callers fall back to LLM generation. Otherwise delegates to
    ``select_language_content`` (``fallback="warn"``).
    """
    if not text:
        return None
    if not _LANG_MARKER_RE.search(text):
        return None
    return select_language_content(text, target_lang, fallback="warn")


def process_text(text: str, machine: Any, lang: str) -> str:
    """DEPRECATED — use ``select_language_content`` + ``render_template``.

    Filters language markers via the consolidated
    ``select_language_content`` (one fewer hop than the previous
    ``filter_language_markers`` indirection) and renders Jinja2
    templates.
    """
    text = select_language_content(text, lang, fallback="warn")
    text = render_template(text, machine)
    return text


# ------------------------------------------------------------------
# compose_system_prompt_for_node — pure function (from dograh pattern)
# ------------------------------------------------------------------


def compose_system_prompt_for_node(
    node_instruction: str,
    system_prompt: str = "",
    *,
    node_type: str = "instruction",
    is_final: bool = False,
    language: str = "en",
) -> str:
    """Pure function: build system prompt for a node.

    Ported from dograh's clean composition pattern. Separates
    identity (who) from task (what to do now).

    Args:
        node_instruction: Per-node enriched instruction text.
        system_prompt: Flow-level system prompt (persona/identity).
        node_type: One of 'final', 'static', 'instruction', 'router'.
        is_final: Whether this is the final node.
        language: Target language code.
    """
    sections: list[str] = []

    # Layer 1: Identity (persona from flow)
    if system_prompt:
        sections.append(system_prompt.strip())

    # Layer 2: Task (per-node instruction)
    if node_instruction:
        sections.append(node_instruction.strip())

    # Layer 3: Flow control (per node type)
    control = _flow_control_for_type(node_type)
    if control:
        sections.append(control)

    # Layer 4: Language directive
    if language and language != "en":
        lang_name = {"hi": "Hindi", "es": "Spanish"}.get(language, language)
        sections.append(f"Speak in {lang_name}.")

    # Layer 5: Final node directive
    if is_final:
        sections.append(
            "This is the final step. After delivering your "
            "message, call end_call to disconnect."
        )

    return "\n\n".join(sections)


def _flow_control_for_type(node_type: str) -> str:
    """Minimal flow control instruction per node type."""
    if node_type == "final":
        return "Deliver the final message, then call end_call."
    if node_type == "static":
        return (
            "The message has been spoken. Listen and route using the available tools."
        )
    if node_type == "router":
        return "Decide which path to take. Call the appropriate tool."
    # instruction (default)
    return (
        "Have the conversation naturally. "
        "When ready to proceed, call the appropriate tool."
    )


# ------------------------------------------------------------------
# InstructionComposer (existing — kept for backward compat)
# ------------------------------------------------------------------


class InstructionComposer:
    """Config-driven instruction builder for flow agents.

    Assembles:
    1. Voice preamble (VOICE_CORE + LEADERSHIP + TONE)
    2. System prompt identity (adaptive ~4K)
    3. Per-node enriched instructions (from machine)
    4. Language utterance directive
    5. Timing directive
    6. Final-node directive
    """

    def __init__(
        self,
        machine: Any,
        *,
        lang: str = "en",
        tone: str = "professional",
        include_preamble: bool = True,
        include_system_prompt: bool = True,
        include_leadership: bool = True,
    ) -> None:
        self._machine = machine
        self._lang = lang
        self._tone = tone
        self._include_preamble = include_preamble
        self._include_system_prompt = include_system_prompt
        self._include_leadership = include_leadership

    def compose(
        self,
        enriched: str,
        system_prompt: str = "",
        *,
        is_final: bool = False,
    ) -> str:
        """Compose full LLM instructions for a node."""
        sections: list[str] = []

        if self._include_preamble:
            identity = extract_identity(system_prompt) if system_prompt else ""
            preamble = build_voice_preamble(
                self._tone,
                identity,
                include_leadership=self._include_leadership,
            )
            sections.append(preamble)
        elif self._include_system_prompt and system_prompt:
            sections.append(system_prompt.strip())

        sections.append(TIMING_DIRECTIVE)

        if enriched:
            processed = process_text(enriched, self._machine, self._lang)
            sections.append(processed)

        sections.append(LANGUAGE_UTTERANCE_DIRECTIVE)

        if is_final:
            sections.append(
                "[IMPORTANT] This is the final step of the conversation. "
                "After delivering your message, you MUST call the end_call "
                "tool to disconnect. Do NOT continue the conversation. "
                "If the user speaks again, briefly acknowledge and call "
                "end_call immediately."
            )

        return "\n\n".join(sections)
