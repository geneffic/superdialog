"""Inlined replacements for super.core.voice.common.* and managers.prompt_manager.

These three utilities were single-purpose imports from the parent codebase.
Inlining them removes the cross-package dependency for the OSS build.
The original production code can monkey-patch save_failed_execution_log if
it needs real failure logging.
"""

from __future__ import annotations

from typing import Any


def detect_language(text: str) -> str:
    """Naive heuristic: Devanagari range -> 'hi', else 'en'.

    Limitations:
        - Only covers Devanagari proper (U+0900-U+097F). Other scripts
          (Tamil, Bengali, Arabic, CJK, etc.) all fall through to 'en'.
        - No statistical detection, no script-mixing handling.

    Production deployments should monkey-patch this reference to plug
    in a serious language detector (e.g. fasttext, langid).
    """
    if not text:
        return "en"
    for ch in text:
        if "ऀ" <= ch <= "ॿ":
            return "hi"
    return "en"


_LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}


def get_language_name(code: str) -> str:
    """Return human-readable name for a language code (fallback: code)."""
    return _LANG_NAMES.get(code, code)


async def save_failed_execution_log(*args: Any, **kwargs: Any) -> None:
    """No-op in OSS build. Production runtime can monkey-patch this."""
    return None
