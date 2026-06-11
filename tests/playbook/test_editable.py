"""Tests for the EditableDoc abstraction (FullDoc / SimpleDoc)."""

import pytest

from superdialog.playbook.editable import Edit, FullDoc, MutationError
from tests.playbook.test_models import MINIMAL_YAML

_GUIDANCE = "journeys.booking.checkpoints.collect.guidance"


def test_fields_enumerates_exactly_the_whitelist() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    addrs = {f.address for f in doc.fields()}
    assert "persona" in addrs
    assert _GUIDANCE in addrs
    assert "journeys.booking.checkpoints.collect.goal" in addrs
    assert "journeys.booking.checkpoints.collect.slots.city.description" in addrs
    # the collect rule is llm-judged -> editable
    assert "journeys.booking.checkpoints.collect.advance_when[0].when" in addrs
    # confirm's rules are expr-judged -> frozen
    assert "journeys.booking.checkpoints.confirm.advance_when[0].when" not in addrs
    # say_verbatim editable only where present
    assert "journeys.booking.checkpoints.confirm.say_verbatim" in addrs
    assert "journeys.booking.checkpoints.collect.say_verbatim" not in addrs
    # structure is unreachable
    assert "journeys.booking.checkpoints.confirm.gate" not in addrs


def test_apply_returns_new_doc_and_compiles() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    edited = doc.apply([Edit(address=_GUIDANCE, new_text="Collect warmly.")])
    assert edited.compile().checkpoint("booking.collect").guidance == "Collect warmly."
    # the original is untouched (apply is functional)
    assert doc.compile().checkpoint("booking.collect").guidance == "Collect naturally."


def test_emit_diff_touches_only_the_edited_line() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    edited = doc.apply([Edit(address=_GUIDANCE, new_text="Collect warmly.")])
    before = doc.emit().splitlines()
    after = edited.emit().splitlines()
    assert len(before) == len(after)
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(changed) == 1
    assert "Collect warmly." in changed[0][1]


def test_apply_rejects_non_whitelisted_addresses() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    for bad in (
        "journeys.booking.checkpoints.confirm.gate",  # structure
        "journeys.booking.checkpoints.confirm.advance_when[0].when",  # expr
        "journeys.booking.checkpoints.collect.say_verbatim",  # absent -> no add
        "journeys.booking.checkpoints.nope.guidance",  # unknown checkpoint
        "tools",  # structure
    ):
        with pytest.raises(MutationError):
            doc.apply([Edit(address=bad, new_text="x")])


def test_never_say_entries_may_be_added_but_not_removed() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    addr = "journeys.booking.checkpoints.collect.never_say"
    grown = doc.apply([Edit(address=addr, new_text=["never promise refunds"])])
    cp = grown.compile().checkpoint("booking.collect")
    assert cp.never_say == ["never promise refunds"]
    with pytest.raises(MutationError):
        grown.apply([Edit(address=addr, new_text=[])])  # shrinking is removal


def test_string_field_requires_string_payload() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    with pytest.raises(MutationError):
        doc.apply([Edit(address=_GUIDANCE, new_text=["not", "a", "string"])])


def test_pipeline_on_keys_survive_the_round_trip() -> None:
    # MINIMAL_YAML's pipeline uses an `on:` key; YAML 1.1 would load it as a
    # boolean. FullDoc must parse with the models loader, not yaml.safe_load.
    doc = FullDoc.from_text(MINIMAL_YAML)
    reparsed = FullDoc.from_text(doc.emit())
    assert reparsed.compile().pipeline("confirm_and_hold").steps[0].on
