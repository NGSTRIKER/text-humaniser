"""Repeated vocabulary.

Fixes accidental doubled words ("the the").  Overused content words and
repeated sentence openings are *flagged* (replacement=None) so they show
up in the edit history; the synonym rule (Stage 4) proposes alternatives.
"""

from __future__ import annotations

import re

from humanizer.candidates import Candidate
from humanizer.config import Intensity
from humanizer.transformations.base import Rule, RuleContext

# Doubled words that can be grammatical: "had had", "that that", "is is" (rare).
LEGIT_DOUBLES = frozenset({"had", "that", "is", "do", "very", "so", "no", "bye", "ha", "knock"})


class DoubledWordRule(Rule):
    rule_id = "repetition.doubled_word"
    description = "Remove an accidentally doubled word."
    transform_type = "repetition"
    min_intensity = Intensity.LIGHT

    def propose(self, ctx: RuleContext) -> list[Candidate]:
        out = []
        for m in re.finditer(r"\b(\w+)(\s+)(\1)\b", ctx.text, re.IGNORECASE):
            word = m.group(1)
            if word.lower() in LEGIT_DOUBLES or word.isdigit():
                continue
            # replace "the the" with the first word only
            out.append(
                Candidate(
                    rule_id=self.rule_id, transform_type=self.transform_type,
                    char_start=m.start(), char_end=m.end(), original=m.group(0),
                    replacement=word,
                    reason=f"Removed accidentally doubled word {word!r}.",
                    confidence=0.95, meaning_risk=0.02, grammar_risk=0.02,
                    usefulness=0.9, diversity_gain=0.3, priority=3,
                )
            )
        return out


class RepeatedVocabularyRule(Rule):
    rule_id = "repetition.overused_word"
    description = "Flag content words repeated within a sentence or across nearby sentences."
    transform_type = "repetition"
    min_intensity = Intensity.LIGHT

    def propose(self, ctx: RuleContext) -> list[Candidate]:
        a = ctx.analysis
        proper = {e.text.lower() for e in a.entities}
        out: list[Candidate] = []
        for sent in a.sentences:
            seen: dict[str, int] = {}
            for ti in sent.token_indices:
                tok = a.tokens[ti]
                if tok.is_stop or tok.is_punct or tok.pos not in {"NOUN", "VERB", "ADJ", "ADV"}:
                    continue
                if tok.text.lower() in proper:
                    continue
                lemma = tok.lemma.lower()
                if lemma in seen:
                    out.append(
                        Candidate(
                            rule_id=self.rule_id, transform_type=self.transform_type,
                            char_start=tok.char_start, char_end=tok.char_end,
                            original=tok.text, replacement=None,
                            reason=f"{tok.text!r} repeats within one sentence.",
                            confidence=0.6, meaning_risk=0.3, diversity_gain=0.5,
                            note="detection only; see the synonym rule for alternatives",
                        )
                    )
                else:
                    seen[lemma] = ti
        return out


class RepeatedOpeningRule(Rule):
    rule_id = "repetition.sentence_opening"
    description = "Flag runs of sentences that start with the same words."
    transform_type = "repetition"
    min_intensity = Intensity.LIGHT

    def propose(self, ctx: RuleContext) -> list[Candidate]:
        out = []
        sents = ctx.analysis.sentences
        for prev, cur in zip(sents, sents[1:]):
            if cur.opening and cur.opening.lower() == prev.opening.lower():
                start = cur.char_start
                out.append(
                    Candidate(
                        rule_id=self.rule_id, transform_type=self.transform_type,
                        char_start=start, char_end=start + len(cur.opening),
                        original=cur.opening, replacement=None,
                        reason=f"Consecutive sentences both open with {cur.opening!r}.",
                        confidence=0.7, meaning_risk=0.3, diversity_gain=0.6,
                        note="sentence-variety rewrites arrive in Stage 5",
                    )
                )
        return out
