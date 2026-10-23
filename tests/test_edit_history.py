"""Edit application, undo and explanation tests (no spaCy needed)."""

from __future__ import annotations

import pytest

from humanizer.edit_history import EditHistory, apply_edits
from humanizer.schemas import EditRecord


def rec(text, original, replacement, eid="e1", ttype="wordiness"):
    start = text.index(original)
    return EditRecord(edit_id=eid, original_span=original, replacement=replacement,
                      rule_id="r", transform_type=ttype, accepted=True,
                      char_start=start, char_end=start + len(original))


def test_multiple_edits_keep_offsets_valid():
    text = "We left due to the fact that the end result was bad."
    edits = [rec(text, "due to the fact that", "because", "e1"),
             rec(text, "end result", "result", "e2")]
    assert apply_edits(text, edits) == "We left because the result was bad."


def test_deletion_at_sentence_start_recapitalises():
    text = "It is important to note that sales rose."
    assert apply_edits(text, [rec(text, "It is important to note that", "")]) == "Sales rose."


def test_deletion_mid_sentence_cleans_spaces():
    text = "It was very slow."
    assert apply_edits(text, [rec(text, "very", "")]) == "It was slow."


def test_article_agreement():
    text = "It was an unexpected surprise."
    assert apply_edits(text, [rec(text, "unexpected surprise", "surprise")]) == "It was a surprise."


def test_stale_edit_raises():
    text = "abc def"
    bad = EditRecord(edit_id="x", original_span="zzz", replacement="", char_start=0, char_end=3)
    with pytest.raises(ValueError):
        apply_edits(text, [bad])


def test_undo_and_redo():
    text = "We left due to the fact that the end result was bad."
    h = EditHistory(text, [rec(text, "due to the fact that", "because", "e1"),
                           rec(text, "end result", "result", "e2")])
    assert h.undo("e2") == "We left because the end result was bad."
    assert h.redo("e2") == "We left because the result was bad."


def test_explain_lists_changes_and_skips():
    text = "The end result."
    accepted = rec(text, "end result", "result", "e1", "redundancy")
    skipped = EditRecord(edit_id="e2", original_span="The", replacement="", rule_id="x",
                         accepted=False, rejection_reason="flag only", char_start=0, char_end=3)
    h = EditHistory(text, [accepted, skipped])
    out = h.explain(include_rejected=True)
    assert "Removed redundant wording" in out
    assert "flag only" in out
