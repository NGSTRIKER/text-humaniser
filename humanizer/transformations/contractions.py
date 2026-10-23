"""Expand contractions for professional and academic styles.

Only unambiguous forms are expanded.  "'s" (is/has/possessive) and "'d"
(had/would) are left alone because the right expansion depends on context.
"""

from __future__ import annotations

from humanizer.candidates import Candidate
from humanizer.config import Intensity
from humanizer.transformations.base import Rule, RuleContext, match_case

EXPANSIONS: dict[str, str] = {
    "can't": "cannot", "won't": "will not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "shouldn't": "should not", "wouldn't": "would not", "couldn't": "could not",
    "mustn't": "must not", "i'm": "I am", "you're": "you are", "we're": "we are",
    "they're": "they are", "i've": "I have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "I will", "you'll": "you will", "we'll": "we will",
    "they'll": "they will", "it'll": "it will", "let's": "let us",
}


class ContractionRule(Rule):
    rule_id = "style.contractions"
    description = "Expand contractions (professional/academic styles only)."
    transform_type = "style"
    min_intensity = Intensity.LIGHT

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def propose(self, ctx: RuleContext) -> list[Candidate]:
        if not self.enabled:
            return []
        pattern = "|".join(k.replace("'", "['’]") for k in EXPANSIONS)
        out = []
        for m in ctx.finditer(pattern):
            orig = m.group(0)
            full = EXPANSIONS[orig.lower().replace("’", "'")]
            rep = full if full.startswith("I ") else match_case(orig, full)
            out.append(Candidate(
                rule_id=self.rule_id, transform_type=self.transform_type,
                char_start=m.start(), char_end=m.end(), original=orig, replacement=rep,
                reason=f"Formal style: expanded {orig!r} to {rep!r}.",
                confidence=0.95, meaning_risk=0.0, grammar_risk=0.02, usefulness=0.6, priority=1,
            ))
        return out
