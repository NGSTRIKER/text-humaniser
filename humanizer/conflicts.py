"""Edit-conflict detection (Stage 6, plan §8: "conflicts with previously selected edits").

Two acceptable candidates conflict when applying both could produce an
invalid or unnatural result even though each is fine alone:

* ``overlap``    — their character spans intersect (offsets would break).
* ``adjacent``   — same transform type touching each other ("very very").
* ``structural`` — two whole-sentence restructurings in one sentence.
* ``echo``       — two replacements introduce the same content word close
  together, creating the repetition the rules are meant to remove.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from humanizer.candidates import Candidate
from humanizer.profiles import STRUCTURAL_TYPES

ECHO_WINDOW = 120  # characters
_WORD = re.compile(r"[A-Za-z']{4,}")


def _sentence_of(pos: int, bounds: Sequence[tuple[int, int]]) -> int:
    for i, (s, e) in enumerate(bounds):
        if s <= pos < e:
            return i
    return -1


def _new_words(c: Candidate) -> set[str]:
    if not c.replacement:
        return set()
    before = {w.lower() for w in _WORD.findall(c.original)}
    return {w.lower() for w in _WORD.findall(c.replacement)} - before


def conflict(
    a: Candidate, b: Candidate, sentence_bounds: Sequence[tuple[int, int]] = ()
) -> Optional[str]:
    """Return the conflict kind between ``a`` and ``b`` or ``None``."""
    if a.overlaps(b):
        return "overlap"
    if a.transform_type == b.transform_type and (
        a.char_end == b.char_start or b.char_end == a.char_start
    ):
        return "adjacent"
    if (
        sentence_bounds
        and a.transform_type in STRUCTURAL_TYPES
        and b.transform_type in STRUCTURAL_TYPES
        and _sentence_of(a.char_start, sentence_bounds) == _sentence_of(b.char_start, sentence_bounds) != -1
    ):
        return "structural"
    gap = max(a.char_start, b.char_start) - min(a.char_end, b.char_end)
    if gap <= ECHO_WINDOW and _new_words(a) & _new_words(b):
        return "echo"
    return None


_MESSAGES = {
    "overlap": "overlaps selected edit {}",
    "adjacent": "adjacent to selected edit {} of the same type",
    "structural": "sentence already restructured by {}",
    "echo": "would repeat a word introduced by {}",
}


def conflict_message(kind: str, rule_id: str) -> str:
    return _MESSAGES[kind].format(rule_id)
