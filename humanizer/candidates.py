"""Candidate edits proposed by rules, plus protection helpers.

A :class:`Candidate` is a *proposal*.  Nothing is changed until the
weighting/selection step accepts it and the edit history applies it.
All offsets refer to the ORIGINAL text passed to the rule engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

from humanizer.analyzer import DocumentAnalysis


@lru_cache(maxsize=256)
def _term_regex(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


@dataclass
class Candidate:
    """One proposed edit.

    Attributes:
        rule_id: Unique id of the rule that proposed it (e.g. ``wordiness.in_order_to``).
        transform_type: Category: redundancy | wordiness | repetition | phrase.
        char_start / char_end: Span in the original text.
        original: Text currently in that span.
        replacement: Proposed text; ``None`` means "flag only, no safe fix".
        reason: Human-readable explanation.
        confidence: How sure the rule is that the edit is correct (0–1).
        meaning_risk: Estimated chance the edit changes meaning (0–1).
        grammar_risk: Estimated chance the edit breaks grammar (0–1).
        usefulness: How much the edit improves the text (0–1).
        context_fit: How well the replacement fits its surroundings (0–1).
        diversity_gain: How much it reduces repetition / monotony (0–1).
        grammar_ok: Result of the rule's own grammatical compatibility check.
        priority: Higher = more important when tie-breaking.
        note: Extra detail (e.g. why the grammar check failed).
    """

    rule_id: str
    transform_type: str
    char_start: int
    char_end: int
    original: str
    replacement: Optional[str]
    reason: str
    confidence: float = 0.8
    meaning_risk: float = 0.1
    grammar_risk: float = 0.1
    usefulness: float = 0.6
    context_fit: float = 0.8
    diversity_gain: float = 0.0
    grammar_ok: bool = True
    priority: int = 1
    note: str = ""
    score: float = 0.0
    # Structural edits (voice, reordering) rewrite whole sentences.  They may
    # contain protected spans as long as each one appears unchanged in the
    # replacement.
    preserves_protected: bool = False
    # Stage 6: risk of mangling domain/technical terminology (0–1) and the
    # probability this candidate had when it was drawn.
    domain_risk: float = 0.0
    selection_probability: float | None = None

    def overlaps(self, other: "Candidate") -> bool:
        return self.char_start < other.char_end and other.char_start < self.char_end

    def overlaps_span(self, start: int, end: int) -> bool:
        return self.char_start < end and start < self.char_end

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


# Entity labels that must never be touched by rules.
PROTECTED_ENTITY_LABELS = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "NORP", "FAC", "EVENT",
     "WORK_OF_ART", "LAW", "LANGUAGE", "MONEY", "PERCENT", "QUANTITY"}
)


def protected_spans(
    analysis: DocumentAnalysis, protected_terms: Iterable[str] = ()
) -> list[tuple[int, int, str]]:
    """Spans no edit may overlap: names, numbers, negation, user terms.

    Returns a list of ``(start, end, why)`` tuples.
    """
    spans: list[tuple[int, int, str]] = []
    for ent in analysis.entities:
        if ent.label in PROTECTED_ENTITY_LABELS:
            spans.append((ent.char_start, ent.char_end, f"named entity ({ent.label})"))
    for q in analysis.quantities:
        if any(ch.isdigit() for ch in q.text):
            spans.append((q.char_start, q.char_end, f"number/{q.kind}"))
    for neg in analysis.negations:
        spans.append((neg.char_start, neg.char_end, "negation"))
    for term in protected_terms:
        term = term.strip()
        if not term:
            continue
        for m in _term_regex(term).finditer(analysis.text):
            spans.append((m.start(), m.end(), "protected term"))
    return spans


def protection_conflict(
    cand: Candidate, spans: list[tuple[int, int, str]]
) -> Optional[str]:
    """Return why ``cand`` touches a protected span, or ``None``."""
    for start, end, why in spans:
        if not cand.overlaps_span(start, end):
            continue
        inside = cand.char_start <= start and end <= cand.char_end
        if cand.preserves_protected and inside and cand.replacement is not None:
            piece = cand.original[start - cand.char_start : end - cand.char_start]
            if piece.lower() in cand.replacement.lower():
                continue
            return f"structural edit lost protected {why}: {piece!r}"
        if (cand.transform_type == "style" and "negation" in why
                and re.search(r"\bnot\b|\bcannot\b", cand.replacement or "", re.I)):
            continue  # "don't" -> "do not" keeps the negation
        return f"touches protected {why}: {cand.original!r}"
    return None
