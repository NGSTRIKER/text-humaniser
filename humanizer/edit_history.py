"""Edit tracking, application, explanation and undo.

Edits are stored against ORIGINAL offsets and applied right-to-left, so
offsets never go stale.  Undoing an edit simply re-applies the remaining
accepted edits to the original text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from humanizer.schemas import EditRecord

VOWEL_SOUND = re.compile(r"^(?:[aeio]|u(?![nr]i|se|sa|ti|ni)|hour|honest|honou?r)", re.IGNORECASE)

TYPE_LABELS = {
    "redundancy": "Removed redundant wording",
    "wordiness": "Simplified wordy phrase",
    "repetition": "Reduced repetition",
    "phrase": "Restructured a phrase",
    "synonym": "Reduced repeated vocabulary",
    "voice": "Changed voice",
    "word_order": "Rearranged a phrase",
    "sentence_structure": "Split or combined sentences",
    "paragraph_flow": "Reordered sentences",
}


def apply_edits(text: str, edits: Iterable[EditRecord]) -> str:
    """Apply non-overlapping edits (original offsets) and tidy the seams."""
    ordered = sorted(edits, key=lambda e: e.char_start, reverse=True)
    for e in ordered:
        if text[e.char_start : e.char_end] != e.original_span:
            raise ValueError(f"edit {e.edit_id} no longer matches the text")
        text = _splice(text, e.char_start, e.char_end, e.replacement)
    return text


def _splice(text: str, start: int, end: int, replacement: str) -> str:
    before, after = text[:start], text[end:]
    at_sentence_start = before.rstrip() == "" or before.rstrip()[-1] in ".!?"
    was_capitalised = text[start : start + 1].isupper()

    if replacement == "":
        # remove one side of the surrounding whitespace
        if before.endswith(" ") and (after.startswith(" ") or after[:1] in {",", ".", ";", ":", "!", "?", ""}):
            before = before[:-1]
        elif after.startswith(" ") and (before == "" or before.endswith(("\n", " "))):
            after = after[1:]
        # drop a now-dangling comma at clause start: "It should be noted that, X"
        if at_sentence_start and after.startswith(", "):
            after = after[2:]
        if at_sentence_start and was_capitalised:
            after = re.sub(r"^(\s*)(\w)", lambda m: m.group(1) + m.group(2).upper(), after, count=1)
        out = before + after
    else:
        out = before + replacement + after

    return _fix_article(out, len(before))


def _fix_article(text: str, pos: int) -> str:
    """Fix 'a'/'an' immediately before ``pos`` to agree with the next word."""
    m = re.search(r"(?<!\w)(a|an|A|An)(\s+)$", text[:pos])
    if not m:
        return text
    nxt = re.match(r"\s*(\w+)", text[pos:])
    if not nxt:
        return text
    want_an = bool(VOWEL_SOUND.match(nxt.group(1)))
    art = m.group(1)
    new = ("an" if want_an else "a")
    if art[0].isupper():
        new = new.capitalize()
    if new == art:
        return text
    return text[: m.start(1)] + new + text[m.end(1) :]


@dataclass
class EditHistory:
    """All proposed edits for one text, with accepted/rejected status."""

    original_text: str
    records: list[EditRecord] = field(default_factory=list)

    @property
    def accepted(self) -> list[EditRecord]:
        return [r for r in self.records if r.accepted and r.user_decision is not False]

    @property
    def rejected(self) -> list[EditRecord]:
        return [r for r in self.records if not r.accepted]

    def get(self, edit_id: str) -> Optional[EditRecord]:
        return next((r for r in self.records if r.edit_id == edit_id), None)

    def render(self) -> str:
        """Text with all accepted (and not user-rejected) edits applied."""
        return apply_edits(self.original_text, self.accepted)

    def undo(self, edit_id: str) -> str:
        """Mark an edit as rejected by the user and return the new text."""
        rec = self.get(edit_id)
        if rec is None:
            raise KeyError(edit_id)
        rec.user_decision = False
        return self.render()

    def redo(self, edit_id: str) -> str:
        rec = self.get(edit_id)
        if rec is None:
            raise KeyError(edit_id)
        rec.user_decision = None
        return self.render()

    def explain(self, include_rejected: bool = False) -> str:
        """Human-readable list of changes."""
        lines = []
        for r in sorted(self.accepted, key=lambda r: r.char_start):
            label = TYPE_LABELS.get(r.transform_type, "Changed")
            arrow = f"{r.replacement!r}" if r.replacement else "(removed)"
            lines.append(
                f"[{r.edit_id}] {label}: {r.original_span!r} → {arrow}  "
                f"(rule {r.rule_id}, confidence {r.confidence:.2f}, risk {r.semantic_risk:.2f})"
            )
        if include_rejected:
            for r in sorted(self.rejected, key=lambda r: r.char_start):
                lines.append(f"[{r.edit_id}] Skipped {r.original_span!r} ({r.rule_id}): {r.rejection_reason}")
        return "\n".join(lines) if lines else "No changes."
