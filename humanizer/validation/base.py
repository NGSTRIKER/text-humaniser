"""Shared types and text helpers for the validation checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

from humanizer.analyzer import DocumentAnalysis, Span


class Severity(str, Enum):
    CRITICAL = "critical"   # meaning change → never auto-accept
    MAJOR = "major"         # likely problem → reject under medium/high strictness
    MINOR = "minor"         # worth reporting; does not block


@dataclass
class Issue:
    category: str            # e.g. number, negation, role, addition, grammar
    severity: Severity
    message: str
    original: Optional[str] = None
    rewritten: Optional[str] = None
    repair: Optional[str] = None   # suggested repair instruction for the LLM


@dataclass
class CheckResult:
    name: str
    issues: list[Issue] = field(default_factory=list)
    score: Optional[float] = None     # 0–1 where meaningful
    skipped: bool = False
    note: str = ""

    @property
    def passed(self) -> bool:
        return not any(i.severity == Severity.CRITICAL for i in self.issues)

    def add(self, *args: Any, **kw: Any) -> None:
        self.issues.append(Issue(*args, **kw))


@dataclass
class Pair:
    """Original and rewritten text with their analyses."""

    original: DocumentAnalysis
    rewritten: DocumentAnalysis


DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "its", "their", "his", "her", "our", "my", "your"}


def norm_phrase(text: str) -> str:
    words = [w.strip(".,;:!?\"'()").lower() for w in text.split()]
    return " ".join(w for w in words if w and w not in DETERMINERS)


def span_key(span: Optional[Span], doc: DocumentAnalysis) -> Optional[str]:
    """Normalized lemma head of a span ("the big rocks" → "rock")."""
    if span is None:
        return None
    toks = [t for t in doc.tokens if span.char_start <= t.char_start < span.char_end and not t.is_punct]
    content = [t for t in toks if t.pos in {"NOUN", "PROPN", "PRON", "NUM"}]
    if content:
        # the syntactic head is the content token whose head lies outside the span
        idx = {t.index for t in toks}
        heads = [t for t in content if t.head_index not in idx] or content[-1:]
        return heads[0].lemma.lower()
    return norm_phrase(span.text) or None


@lru_cache(maxsize=1)
def _graph() -> Any:
    try:
        from humanizer.word_graph import get_graph, is_available
        return get_graph() if is_available() else None
    except Exception:  # pragma: no cover - optional dependency
        return None


@lru_cache(maxsize=8192)
def related(a: str, b: str, pos: Optional[str] = None) -> bool:
    """True if ``a`` and ``b`` are the same word or close WordNet relatives (not antonyms)."""
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    g = _graph()
    if g is None:
        return False
    from humanizer.word_graph import Relation
    # Broader/narrower terms count as counterparts ("business" ~ "firm",
    # "grow" ~ "expand"); antonyms never do.
    good = {Relation.SYNONYM, Relation.SIMILAR, Relation.DERIVED, Relation.BROADER, Relation.NARROWER}
    return bool((g.relation_between(a, b, pos) | g.relation_between(b, a, pos)) & good)


def dispensable_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges that rules may legitimately delete (filler and
    redundant modifiers such as "in order", "end" in "end result")."""
    import re
    from humanizer.analyzer import FILLER_PHRASES
    from humanizer.transformations.redundancy import PATTERNS as REDUNDANCY_PATTERNS
    from humanizer.transformations.wordiness import PATTERNS as WORDINESS_PATTERNS
    from humanizer.transformations.phrases import ITEMS as PHRASE_ITEMS

    out: list[tuple[int, int]] = []
    low = text.lower()
    for phrase in FILLER_PHRASES:
        for m in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", low):
            out.append((m.start(), m.end()))
    for p in REDUNDANCY_PATTERNS:
        for m in re.finditer(r"(?<!\w)" + p.pattern + r"(?!\w)", low):
            if m.start() < m.start(1):
                out.append((m.start(), m.start(1)))
            if m.end(1) < m.end():
                out.append((m.end(1), m.end()))
    for wp in WORDINESS_PATTERNS:
        for m in re.finditer(r"(?<!\w)(?:" + wp.pattern + r")(?!\w)", low):
            out.append((m.start(), m.end()))
    for item in PHRASE_ITEMS:
        verbs = "|".join(sorted(set(item.forms), key=len, reverse=True))
        for m in re.finditer(r"(?<!\w)(?:" + verbs + r") " + item.tail + r"(?!\w)", low):
            out.append((m.start(), m.end()))
    return out
