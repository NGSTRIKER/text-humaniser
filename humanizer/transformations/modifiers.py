"""Modifier management: remove unnecessary adjectives, adverbs, intensifiers, and pleonasms.

Follows Stage 3 / prompt §5.E:
- Pleonastic modifier-head pairs (e.g. "shouted loudly" → "shouted", "completely eliminate" → "eliminate")
- Empty intensifiers and weak adverbs modifying adjectives/adverbs (e.g. "very good" → "good")
- Weak fillers modifying verbs (e.g. "basically ran" → "ran")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from humanizer.candidates import Candidate
from humanizer.config import Intensity
from humanizer.transformations.base import INTENSITY_ORDER, Rule, RuleContext, match_case


@dataclass(frozen=True)
class ModifierPattern:
    key: str
    pattern: str  # regex matching the full phrase; group 1 is what to KEEP
    meaning_risk: float = 0.05
    confidence: float = 0.85
    usefulness: float = 0.6
    min_intensity: Intensity = Intensity.LIGHT


# Pleonasms: where the modifier is completely implied by the head word.
PLEONASMS: tuple[ModifierPattern, ...] = (
    # Adverb + Verb pleonasms
    ModifierPattern("shout_loudly", r"(?:shout|shouts|shouted|shouting) (loudly)", 0.05, 0.9, 0.7),
    ModifierPattern("whisper_softly", r"(?:whisper|whispers|whispered|whispering) (softly|quietly)", 0.05, 0.9, 0.7),
    ModifierPattern("sprint_quickly", r"(?:sprint|sprints|sprinted|sprinting) (quickly|fast)", 0.05, 0.9, 0.7),
    ModifierPattern("hurry_quickly", r"(?:hurry|hurries|hurried|hurrying) (quickly|fast)", 0.05, 0.9, 0.7),
    ModifierPattern("scrutinize_carefully", r"(?:carefully )?(?:scrutinize|scrutinizes|scrutinized|scrutinizing)(?: carefully)?", 0.05, 0.85, 0.7),
    ModifierPattern("collaborate_jointly", r"(?:collaborate|collaborates|collaborated|collaborating) jointly", 0.05, 0.9, 0.7),
    ModifierPattern("merge_together", r"(?:merge|merges|merged|merging) together", 0.05, 0.9, 0.7),
    ModifierPattern("penetrate_into", r"(?:penetrate|penetrates|penetrated|penetrating) (into)", 0.05, 0.85, 0.6),

    # Adjective + Noun pleonasms
    ModifierPattern("completely_empty", r"completely (empty)", 0.05, 0.9, 0.7),
    ModifierPattern("completely_full", r"completely (full)", 0.05, 0.9, 0.7),
    ModifierPattern("completely_unique", r"(?:completely|very|totally|most) (unique)", 0.05, 0.9, 0.8),
    ModifierPattern("blatantly_obvious", r"blatantly (obvious)", 0.08, 0.85, 0.6),
    ModifierPattern("exact_replica", r"exact (replicas?)", 0.05, 0.9, 0.7),
    ModifierPattern("original_source", r"original (sources?)", 0.05, 0.85, 0.6),
    ModifierPattern("brief_moment", r"brief (moments?)", 0.05, 0.85, 0.6),
    ModifierPattern("small_speck", r"small (specks?)", 0.05, 0.85, 0.6),
    ModifierPattern("burning_embers", r"burning (embers?)", 0.05, 0.85, 0.6),
    ModifierPattern("final_conclusion", r"final (conclusions?)", 0.05, 0.9, 0.7),
    ModifierPattern("unexpected_emergency", r"unexpected (emergenc(?:y|ies))", 0.05, 0.85, 0.6),
    ModifierPattern("major_breakthrough", r"major (breakthroughs?)", 0.1, 0.75, 0.5, Intensity.MODERATE),
)

# Empty intensifiers that can be removed when modifying adjectives/adverbs
# without altering factual meaning.
EMPTY_INTENSIFIERS: tuple[str, ...] = (
    "very",
    "really",
    "extremely",
    "quite",
    "rather",
    "somewhat",
    "fairly",
    "utterly",
    "totally",
    "literally",
    "basically",
    "actually",
    "virtually",
    "practically",
    "genuinely",
    "truly",
)


class ModifierRule(Rule):
    """Detects and simplifies pleonastic modifiers and empty intensifiers."""

    rule_id = "modifiers"
    description = "Remove unnecessary or redundant adjectives and adverbs."
    transform_type = "modifier"
    min_intensity = Intensity.LIGHT

    def propose(self, ctx: RuleContext) -> list[Candidate]:
        out: list[Candidate] = []
        out.extend(self._propose_pleonasms(ctx))
        if INTENSITY_ORDER[ctx.intensity] >= INTENSITY_ORDER[Intensity.MODERATE]:
            out.extend(self._propose_empty_intensifiers(ctx))
        return out

    def _propose_pleonasms(self, ctx: RuleContext) -> list[Candidate]:
        out: list[Candidate] = []
        for p in PLEONASMS:
            if INTENSITY_ORDER[ctx.intensity] < INTENSITY_ORDER[p.min_intensity]:
                continue
            for m in ctx.finditer(p.pattern):
                original = m.group(0)
                # If there's a capture group, keep group 1; otherwise remove modifier
                if m.lastindex and m.lastindex >= 1:
                    keep = m.group(1)
                    # Handle cases where pattern matches head + modifier
                    replacement = match_case(original, original.replace(keep, "").strip()) if keep in ("loudly", "softly", "quietly", "quickly", "fast", "into") else match_case(original, keep)
                else:
                    # e.g. carefully scrutinize -> scrutinize
                    clean = re.sub(r"\b(carefully|loudly|softly|jointly)\s+", "", original, flags=re.IGNORECASE)
                    clean = re.sub(r"\s+(carefully|loudly|softly|jointly)\b", "", clean, flags=re.IGNORECASE)
                    replacement = match_case(original, clean.strip())

                if replacement and replacement != original:
                    out.append(
                        Candidate(
                            rule_id=f"modifiers.{p.key}",
                            transform_type=self.transform_type,
                            char_start=m.start(),
                            char_end=m.end(),
                            original=original,
                            replacement=replacement,
                            reason=f"Removed redundant modifier from {original!r}.",
                            confidence=p.confidence,
                            meaning_risk=p.meaning_risk,
                            grammar_risk=0.05,
                            usefulness=p.usefulness,
                            priority=2,
                        )
                    )
        return out

    def _propose_empty_intensifiers(self, ctx: RuleContext) -> list[Candidate]:
        out: list[Candidate] = []
        for tok in ctx.analysis.tokens:
            if tok.lemma.lower() in EMPTY_INTENSIFIERS and tok.pos == "ADV":
                # Ensure it modifies an ADJ, ADV, or VERB as an adverbial modifier
                after = ctx.token_after(tok.char_end)
                if not after or after.pos not in {"ADJ", "ADV", "VERB"}:
                    continue
                # Don't strip if negated or part of fixed expression (e.g. "not very")
                prev = ctx.token_before(tok.char_start)
                if prev and prev.lemma.lower() in {"not", "never", "hardly", "scarcely", "without"}:
                    continue

                # The span to remove: include trailing whitespace
                start = tok.char_start
                end = tok.char_end
                if end < len(ctx.text) and ctx.text[end] == " ":
                    end += 1

                original = ctx.text[start:end]
                out.append(
                    Candidate(
                        rule_id=f"modifiers.empty_{tok.lemma.lower()}",
                        transform_type=self.transform_type,
                        char_start=start,
                        char_end=end,
                        original=original,
                        replacement="",
                        reason=f"Removed empty intensifier {tok.text!r}.",
                        confidence=0.75,
                        meaning_risk=0.1,
                        grammar_risk=0.05,
                        usefulness=0.5,
                        priority=1,
                    )
                )
        return out
