"""Wordiness and filler: long constructions with shorter equivalents.

Each entry has its own grammar check so, e.g., "in order to" is only
shortened when followed by a verb ("in order to win" → "to win").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from humanizer.candidates import Candidate
from humanizer.config import Intensity
from humanizer.transformations.base import INTENSITY_ORDER, Rule, RuleContext, match_case

Check = Callable[[RuleContext, int, int], Optional[str]]  # returns failure note or None


def _next_is_verb(ctx: RuleContext, start: int, end: int) -> Optional[str]:
    tok = ctx.token_after(end)
    if tok and tok.pos in {"VERB", "AUX"} and tok.tag in {"VB", "VBP"}:
        return None
    return "next word is not a base-form verb"


def _next_is_adj_or_adv(ctx: RuleContext, start: int, end: int) -> Optional[str]:
    tok = ctx.token_after(end)
    if tok and tok.pos in {"ADJ", "ADV"}:
        return None
    return "intensifier does not modify an adjective/adverb"


def _is_adverb(ctx: RuleContext, start: int, end: int) -> Optional[str]:
    tok = ctx.token_at(start)
    if tok and tok.pos == "ADV":
        return None
    return "word is not used as an adverb here"


def _present_able(ctx: RuleContext, start: int, end: int) -> Optional[str]:
    word = ctx.text[start:end].split()[0].lower()
    return None if word in {"is", "are", "am"} else "only present tense maps safely to 'can'"


def _standalone_have(ctx: RuleContext, start: int, end: int) -> Optional[str]:
    prev = ctx.token_before(start)
    if prev and (prev.lemma.lower() in {"do", "will", "would", "shall", "may", "might", "could", "to"} or prev.pos == "AUX" or prev.tag == "MD" or prev.text.lower() in {"not", "n't"}):
        return "preceded by auxiliary or modal"
    return None


def _sentence_initial_or_clause(ctx: RuleContext, start: int, end: int) -> Optional[str]:
    before = ctx.text[:start].rstrip()
    if not before or before[-1] in ".!?;:,":
        return None
    return "phrase is not at the start of a clause"


@dataclass(frozen=True)
class WordyPattern:
    key: str
    pattern: str
    replacement: str
    confidence: float = 0.85
    meaning_risk: float = 0.08
    usefulness: float = 0.6
    check: Optional[Check] = None
    min_intensity: Intensity = Intensity.LIGHT


PATTERNS: tuple[WordyPattern, ...] = (
    WordyPattern("due_to_the_fact_that", r"due to the fact that", "because", 0.9, 0.05, 0.8),
    WordyPattern("owing_to_the_fact_that", r"owing to the fact that", "because", 0.9, 0.05, 0.8),
    WordyPattern("in_spite_of_the_fact_that", r"in spite of the fact that|despite the fact that", "although", 0.85, 0.08, 0.7),
    WordyPattern("in_order_to", r"in order to", "to", 0.85, 0.05, 0.6, _next_is_verb),
    WordyPattern("so_as_to", r"so as to", "to", 0.8, 0.08, 0.5, _next_is_verb),
    WordyPattern("at_this_point_in_time", r"at this point in time|at the present time|at this moment in time", "now", 0.85, 0.08, 0.8),
    WordyPattern("in_the_event_that", r"in the event that", "if", 0.85, 0.08, 0.7),
    WordyPattern("for_the_purpose_of", r"for the purpose of", "for", 0.75, 0.12, 0.5),
    WordyPattern("with_regard_to", r"with regard to|with respect to|in regard to|in relation to", "about", 0.7, 0.15, 0.5, None, Intensity.MODERATE),
    WordyPattern("in_the_near_future", r"in the near future", "soon", 0.8, 0.1, 0.6),
    WordyPattern("at_all_times", r"at all times", "always", 0.8, 0.1, 0.5),
    WordyPattern("on_a_daily_basis", r"on a daily basis", "daily", 0.85, 0.05, 0.6),
    WordyPattern("on_a_regular_basis", r"on a regular basis", "regularly", 0.85, 0.05, 0.6),
    WordyPattern("in_a_timely_manner", r"in a timely manner", "promptly", 0.75, 0.12, 0.5),
    WordyPattern("prior_to", r"prior to", "before", 0.8, 0.08, 0.4, None, Intensity.MODERATE),
    WordyPattern("subsequent_to", r"subsequent to", "after", 0.8, 0.08, 0.4, None, Intensity.MODERATE),
    WordyPattern("in_close_proximity_to", r"in close proximity to", "near", 0.85, 0.08, 0.7),
    WordyPattern("a_large_number_of", r"a large number of|a great number of", "many", 0.85, 0.08, 0.6),
    WordyPattern("a_majority_of", r"the majority of|a majority of", "most", 0.75, 0.12, 0.5),
    WordyPattern("has_the_ability_to", r"(?:has|have) the ability to", "can", 0.8, 0.1, 0.6, _standalone_have),
    WordyPattern("is_able_to", r"(?:is|are|am) able to", "can", 0.75, 0.12, 0.5, _present_able, Intensity.MODERATE),
    WordyPattern("until_such_time_as", r"until such time as", "until", 0.85, 0.05, 0.7),
    WordyPattern("in_view_of_the_fact_that", r"in view of the fact that|for the reason that|on the grounds that", "because", 0.9, 0.05, 0.8),
    WordyPattern("find_it_difficult_to", r"find it difficult to", "struggle to", 0.8, 0.08, 0.6, None, Intensity.MODERATE),
    WordyPattern("finds_it_difficult_to", r"finds it difficult to", "struggles to", 0.8, 0.08, 0.6, None, Intensity.MODERATE),
    WordyPattern("found_it_difficult_to", r"found it difficult to", "struggled to", 0.8, 0.08, 0.6, None, Intensity.MODERATE),
    WordyPattern("the_utilization_of", r"the (?:utilization|usage) of", "using", 0.75, 0.1, 0.6, None, Intensity.MODERATE),
    WordyPattern("with_the_exception_of", r"with the exception of", "except for", 0.85, 0.05, 0.6),
    WordyPattern("in_the_majority_of_cases", r"in the majority of cases|in most cases", "usually", 0.75, 0.1, 0.6, None, Intensity.MODERATE),
    WordyPattern("it_is_important_to_note_that", r"it is (?:important|essential|crucial) to note that|it should be noted that|it is worth noting that", "", 0.75, 0.12, 0.7, _sentence_initial_or_clause, Intensity.MODERATE),
    WordyPattern("in_the_process_of", r"in the process of", "", 0.6, 0.2, 0.4, None, Intensity.HEAVY),
    # Intensifiers / fillers: only modifying an adjective/adverb, never at light.
    WordyPattern("very", r"very", "", 0.6, 0.2, 0.3, _next_is_adj_or_adv, Intensity.MODERATE),
    WordyPattern("really", r"really", "", 0.55, 0.25, 0.3, _next_is_adj_or_adv, Intensity.MODERATE),
    WordyPattern("basically", r"basically", "", 0.6, 0.2, 0.4, _is_adverb, Intensity.MODERATE),
    WordyPattern("actually", r"actually", "", 0.5, 0.3, 0.3, _is_adverb, Intensity.HEAVY),
    WordyPattern("quite", r"quite", "", 0.45, 0.35, 0.2, _next_is_adj_or_adv, Intensity.HEAVY),
)


class WordinessRule(Rule):
    rule_id = "wordiness"
    description = "Replace wordy constructions and filler with simpler equivalents."
    transform_type = "wordiness"
    min_intensity = Intensity.LIGHT

    def propose(self, ctx: RuleContext) -> list[Candidate]:
        out: list[Candidate] = []
        for p in PATTERNS:
            if INTENSITY_ORDER[ctx.intensity] < INTENSITY_ORDER[p.min_intensity]:
                continue
            for m in ctx.finditer(p.pattern):
                original = m.group(0)
                note = p.check(ctx, m.start(), m.end()) if p.check else None
                verb = "Removed filler" if p.replacement == "" else "Simplified wordy phrase"
                out.append(
                    Candidate(
                        rule_id=f"wordiness.{p.key}",
                        transform_type=self.transform_type,
                        char_start=m.start(), char_end=m.end(), original=original,
                        replacement=match_case(original, p.replacement),
                        reason=f"{verb}: {original!r}" + (f" → {p.replacement!r}." if p.replacement else "."),
                        confidence=p.confidence, meaning_risk=p.meaning_risk,
                        grammar_risk=0.1 if p.check else 0.05,
                        usefulness=p.usefulness, grammar_ok=note is None, note=note or "",
                        priority=2,
                    )
                )
        return out
