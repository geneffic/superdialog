"""VariableExtractor — out-of-band LLM extraction of declared variables."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Coroutine

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Type alias for the LLM callable used by the extractor
ExtractorLLMFn = Callable[
    [list[dict[str, Any]]],
    Coroutine[Any, Any, str],
]


class ExtractionVariable(BaseModel):
    """Declaration of a variable to extract from conversation."""

    name: str
    type: str = "string"
    prompt: str = ""
    required: bool = False


class VariableExtractor:
    """Out-of-band LLM extraction of declared variables.

    Runs a separate LLM inference at transition time to pull
    declared variables from conversation history. This is
    additive — it fills gaps, doesn't overwrite edge-collected data.
    """

    def __init__(self, llm_fn: ExtractorLLMFn | None = None) -> None:
        self._llm_fn = llm_fn

    async def extract(
        self,
        variables: list[ExtractionVariable],
        conversation_history: list[dict[str, Any]],
        existing_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract variables from conversation history.

        Args:
            variables: What to extract.
            conversation_history: Full dialog so far.
            existing_data: Already-collected data (skip if present).

        Returns:
            Dict of extracted key-value pairs (nulls excluded).
        """
        if not self._llm_fn:
            return {}

        # Filter out variables already collected with valid values
        pending = [
            v
            for v in variables
            if v.name not in existing_data or existing_data.get(v.name) is None
        ]
        if not pending:
            return {}

        prompt = self._build_extraction_prompt(pending)
        history_text = self._format_history(conversation_history)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": history_text},
        ]

        try:
            raw = await self._llm_fn(messages)
            extracted = self._parse_response(raw, pending)
            if extracted:
                logger.info(
                    "[extractor] extracted %d variables: %s",
                    len(extracted),
                    list(extracted.keys()),
                )
            return extracted
        except Exception as exc:
            logger.warning("[extractor] extraction failed: %s", exc)
            return {}

    def _build_extraction_prompt(self, variables: list[ExtractionVariable]) -> str:
        _ist = timezone(timedelta(hours=5, minutes=30))
        today = datetime.now(_ist).strftime("%A, %d %B %Y")
        lines = [
            f"Today's date is {today}. Use this year when resolving"
            " partial dates like '22 April' or 'next Monday'.",
            "Extract the following variables from the conversation below.",
            "Return ONLY a JSON object. Use null for values not found.",
            "",
        ]
        for v in variables:
            desc = v.prompt or f"Extract {v.name}"
            lines.append(f'- "{v.name}" ({v.type}): {desc}')
        return "\n".join(lines)

    def _format_history(self, history: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"{m.get('role', 'unknown').upper()}: {m.get('content', '')}"
            for m in history
            if m.get("content")
        )

    def _parse_response(
        self,
        raw: str,
        variables: list[ExtractionVariable],
    ) -> dict[str, Any]:
        """Parse LLM JSON response, strip nulls."""
        # Handle markdown code blocks
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        text = match.group(1) if match else raw.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug(
                "[extractor] JSON parse failed for: %s",
                text[:200],
            )
            return {}

        if not isinstance(data, dict):
            return {}

        # Only return declared variable names, skip nulls
        valid_names = {v.name for v in variables}
        return {k: v for k, v in data.items() if k in valid_names and v is not None}
