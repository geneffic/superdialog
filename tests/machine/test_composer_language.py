"""GAP-4 regression tests for ``select_language_content`` and shims."""

from __future__ import annotations

import warnings

import pytest

from superdialog.machine.composer import (
    extract_speech_only,
    extract_speech_text,
    filter_language,
    filter_language_markers,
    process_text,
    select_language_content,
)


class TestSelectLanguageContent:
    def test_picks_marker_for_active_language(self):
        text = "[EN] Hello\n[HI] नमस्ते"
        assert select_language_content(text, "en") == "Hello"
        assert select_language_content(text, "hi") == "नमस्ते"

    def test_unmarked_passthrough(self):
        assert select_language_content("plain", "en") == "plain"
        assert select_language_content("plain", "hi") == "plain"

    def test_empty_string(self):
        assert select_language_content("", "en") == ""

    def test_warn_fallback_emits_warning(self):
        with pytest.warns(UserWarning, match="missing marker for language 'hi'"):
            result = select_language_content("[EN] only english", "hi", fallback="warn")
        assert result == "only english"

    def test_english_fallback_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail
            result = select_language_content(
                "[EN] only english", "hi", fallback="english"
            )
        assert result == "only english"

    def test_raise_fallback(self):
        with pytest.raises(ValueError, match="no marker for language 'hi'"):
            select_language_content("[EN] only english", "hi", fallback="raise")

    def test_case_insensitive_language(self):
        assert select_language_content("[EN] hi", "EN") == "hi"
        assert select_language_content("[EN] hi", "en") == "hi"

    def test_language_alias_mapping(self):
        # 'english' / 'hindi' aliases honoured via _LANG_TAG_MAP.
        text = "[EN] Hello\n[HI] नमस्ते"
        assert select_language_content(text, "english") == "Hello"
        assert select_language_content(text, "hindi") == "नमस्ते"

    def test_multiline_marker_content(self):
        text = "[EN] line1\nline2\n[HI] hindi-line"
        assert select_language_content(text, "en") == "line1\nline2"

    def test_raise_lists_available_markers(self):
        with pytest.raises(ValueError, match=r"available=\['en'\]"):
            select_language_content("[EN] x", "hi", fallback="raise")

    def test_three_languages(self):
        text = "[EN] hello\n[HI] नमस्ते\n[ES] hola"
        assert select_language_content(text, "es") == "hola"
        assert select_language_content(text, "hi") == "नमस्ते"
        assert select_language_content(text, "en") == "hello"


class TestDeprecatedShims:
    """Confirm the 5 historical functions still produce sane answers."""

    def test_filter_language_markers_delegates(self):
        assert filter_language_markers("[EN] x\n[HI] y", "hi") == "y"

    def test_filter_language_delegates(self):
        assert filter_language("[EN] x\n[HI] y", "hi") == "y"

    def test_process_text_delegates(self):
        # process_text(text, machine, lang) — machine arg unused when there
        # are no Jinja2 template variables.
        assert process_text("[EN] x\n[HI] y", None, "hi") == "y"

    def test_extract_speech_only_returns_none_with_no_markers(self):
        assert extract_speech_only("plain text", "en") is None

    def test_extract_speech_only_returns_content(self):
        assert extract_speech_only("[EN] hi\n[HI] hi-hi", "hi") == "hi-hi"

    def test_extract_speech_text_returns_none_with_no_markers(self):
        assert extract_speech_text("plain text", None, "en") is None

    def test_extract_speech_text_returns_content(self):
        # No Jinja vars → render_template returns text as-is.
        assert extract_speech_text("[EN] hi\n[HI] hi-hi", None, "hi") == "hi-hi"
