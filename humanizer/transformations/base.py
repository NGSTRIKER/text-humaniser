"""Shared base class and context for rule modules."""

from __future__ import annotations

import bisect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator, Optional

from humanizer.analyzer import DocumentAnalysis, TokenInfo
from humanizer.candidates import Candidate
from humanizer.config import Intensity

INTENSITY_ORDER = {Intensity.LIGHT: 0, Intensity.MODERATE: 1, Intensity.HEAVY: 2}


@lru_cache(maxsize=256)
def _finditer_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)(?:" + pattern + r")(?!\w)", re.IGNORECASE)


@dataclass
class RuleContext:
    """Everything a rule needs to look at the text."""

    text: str
    analysis: DocumentAnalysis
    intensity: Intensity = Intensity.MODERATE
    _starts: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._starts = [t.char_start for t in self.analysis.tokens]

    # token lookups ------------------------------------------------
    def token_at(self, char: int) -> Optional[TokenInfo]:
        """Token whose span contains ``char``."""
        i = bisect.bisect_right(self._starts, char) - 1
        if 0 <= i < len(self.analysis.tokens):
            tok = self.analysis.tokens[i]
            if tok.char_start <= char < tok.char_end:
                return tok
        return None

    def token_after(self, char: int) -> Optional[TokenInfo]:
        """First token starting at or after ``char``."""
        i = bisect.bisect_left(self._starts, char)
        return self.analysis.tokens[i] if i < len(self.analysis.tokens) else None

    def token_before(self, char: int) -> Optional[TokenInfo]:
        """Last token ending at or before ``char``."""
        i = bisect.bisect_left(self._starts, char) - 1
        while i >= 0 and self.analysis.tokens[i].char_end > char:
            i -= 1
        return self.analysis.tokens[i] if i >= 0 else None

    def finditer(self, pattern: str) -> Iterator[re.Match[str]]:
        """Case-insensitive, word-bounded regex search over the text."""
        return _finditer_regex(pattern).finditer(self.text)


class Rule(ABC):
    """A single, independently testable transformation rule.

    Subclasses set ``rule_id``, ``description``, ``transform_type`` and
    ``min_intensity`` and implement :meth:`propose`.
    """

    rule_id: str = "base"
    description: str = ""
    transform_type: str = "other"
    min_intensity: Intensity = Intensity.LIGHT

    def active_for(self, intensity: Intensity) -> bool:
        return INTENSITY_ORDER[intensity] >= INTENSITY_ORDER[self.min_intensity]

    @abstractmethod
    def propose(self, ctx: RuleContext) -> list[Candidate]:
        """Return candidate edits (never modifies the text)."""


def match_case(original: str, replacement: str) -> str:
    """Give ``replacement`` the capitalisation style of ``original``."""
    if not replacement or not original:
        return replacement
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement
