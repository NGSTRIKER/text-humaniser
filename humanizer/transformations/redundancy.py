"""Redundancy removal: phrases that say the same thing twice.

Examples: "end result" → "result", "revert back" → "revert",
"completely eliminate entirely" → "eliminate".
"""

from __future__ import annotations

from dataclasses import dataclass

from humanizer.candidates import Candidate
from humanizer.config import Intensity
from humanizer.transformations.base import Rule, RuleContext, match_case


@dataclass(frozen=True)
class RedundantPattern:
    key: str          # short id used in the rule_id
    pattern: str      # regex; group 1 is what to KEEP
    meaning_risk: float = 0.05
    confidence: float = 0.9
    usefulness: float = 0.7


PATTERNS: tuple[RedundantPattern, ...] = (
    RedundantPattern("completely_x_entirely", r"completely (eliminat\w*|remov\w*|destroy\w*) entirely", 0.05, 0.9, 0.9),
    RedundantPattern("completely_x", r"completely (eliminat\w*|finish\w*|destroy\w*|fill\w*)", 0.1, 0.8),
    RedundantPattern("x_entirely", r"(eliminat\w*) (?:completely|entirely)", 0.1, 0.8),
    RedundantPattern("end_result", r"end (results?)"),
    RedundantPattern("final_outcome", r"final (outcomes?)"),
    RedundantPattern("past_history", r"past (histor(?:y|ies))"),
    RedundantPattern("past_experience", r"past (experiences?)", 0.15, 0.7),
    RedundantPattern("future_plans", r"future (plans?)", 0.1, 0.8),
    RedundantPattern("advance_planning", r"advance (planning)"),
    RedundantPattern("free_gift", r"free (gifts?)"),
    RedundantPattern("added_bonus", r"added (bonus(?:es)?)"),
    RedundantPattern("unexpected_surprise", r"unexpected (surprises?)"),
    RedundantPattern("true_facts", r"true (facts?)"),
    RedundantPattern("basic_fundamentals", r"basic (fundamentals?)"),
    RedundantPattern("new_innovation", r"new (innovations?)"),
    RedundantPattern("joint_collaboration", r"joint (collaborations?)"),
    RedundantPattern("brief_summary", r"brief (summar(?:y|ies))", 0.1, 0.8),
    RedundantPattern("close_proximity", r"close (proximity)"),
    RedundantPattern("absolutely_essential", r"absolutely (essential)", 0.1, 0.85),
    RedundantPattern("each_and_every", r"each and (every)"),
    RedundantPattern("first_and_foremost", r"(first) and foremost", 0.05, 0.85),
    RedundantPattern("revert_back", r"(revert|reverts|reverted|reverting) back"),
    RedundantPattern("return_back", r"(return|returns|returned|returning) back"),
    RedundantPattern("repeat_again", r"(repeat|repeats|repeated|repeating) again", 0.1, 0.85),
    RedundantPattern("atm_machine", r"(ATM)s? machines?"),
    RedundantPattern("pin_number", r"(PIN) numbers?"),
    RedundantPattern("gather_together", r"(gather|gathers|gathered|gathering) together"),
    RedundantPattern("combine_together", r"(combine|combines|combined|combining) together"),
    RedundantPattern("cooperate_together", r"(cooperate|cooperates|cooperated|cooperating) together"),
    RedundantPattern("period_of_time", r"(period|periods) of time"),
    RedundantPattern("necessary_or_needed", r"(necessary|needed) or (?:needed|necessary)"),
    RedundantPattern("collaborate_together", r"(collaborate|collaborates|collaborated|collaborating) together"),
    RedundantPattern("merge_together", r"(merge|merges|merged|merging) together"),
    RedundantPattern("blend_together", r"(blend|blends|blended|blending) together"),
    RedundantPattern("join_together", r"(join|joins|joined|joining) together"),
    RedundantPattern("connect_together", r"(connect|connects|connected|connecting) together"),
    RedundantPattern("plan_in_advance", r"(plan|plans|planned|planning) in advance"),
    RedundantPattern("exact_same", r"exact (same)"),
    RedundantPattern("small_in_size", r"(small) in size"),
    RedundantPattern("large_in_size", r"(large) in size"),
    RedundantPattern("personal_opinion", r"personal (opinions?)"),
    RedundantPattern("sum_total", r"sum (totals?)"),
    RedundantPattern("foreign_imports", r"foreign (imports?)"),
    RedundantPattern("advance_warning", r"advance (warnings?)"),
    RedundantPattern("advance_reservations", r"advance (reservations?)"),
    RedundantPattern("currently_existing", r"currently (existing)"),
    RedundantPattern("empty_space", r"empty (spaces?)"),
    RedundantPattern("filled_to_capacity", r"(filled) to capacity"),
    RedundantPattern("postpone_until_later", r"(postpone|postpones|postponed|postponing) until later"),
    RedundantPattern("recall_back", r"(recall|recalls|recalled|recalling) back"),
    RedundantPattern("reply_back", r"(reply|replies|replied|replying) back"),
    RedundantPattern("retreat_back", r"(retreat|retreats|retreated|retreating) back"),
    RedundantPattern("full_and_complete", r"full and (complete)"),
    RedundantPattern("true_and_accurate", r"true and (accurate)"),
    RedundantPattern("basic_and_fundamental", r"basic and (fundamental)"),
    RedundantPattern("any_and_all", r"any and (all)"),
)


class RedundantPhraseRule(Rule):
    rule_id = "redundancy"
    description = "Remove words that repeat a meaning already expressed."
    transform_type = "redundancy"
    min_intensity = Intensity.LIGHT

    def propose(self, ctx: RuleContext) -> list[Candidate]:
        out: list[Candidate] = []
        for p in PATTERNS:
            for m in ctx.finditer(p.pattern):
                original = m.group(0)
                keep = m.group(1)
                out.append(
                    Candidate(
                        rule_id=f"redundancy.{p.key}",
                        transform_type=self.transform_type,
                        char_start=m.start(), char_end=m.end(),
                        original=original,
                        replacement=keep if keep.isupper() else match_case(original, keep),
                        reason=f"Removed redundant wording: {original!r} already means {keep!r}.",
                        confidence=p.confidence, meaning_risk=p.meaning_risk,
                        grammar_risk=0.05, usefulness=p.usefulness, priority=2,
                    )
                )
        out.extend(self._reason_is_because(ctx))
        out.extend(self._move_forward(ctx))
        return out

    def _move_forward(self, ctx: RuleContext) -> list[Candidate]:
        """'move forward towards/to' → 'move towards/to'."""
        out = []
        for m in ctx.finditer(r"move forward (towards|to)"):
            original = m.group(0)
            target = f"move {m.group(1)}"
            out.append(
                Candidate(
                    rule_id="redundancy.move_forward",
                    transform_type=self.transform_type,
                    char_start=m.start(), char_end=m.end(), original=original,
                    replacement=match_case(original, target),
                    reason=f"{original!r} repeats direction; {target!r} is cleaner.",
                    confidence=0.85, meaning_risk=0.05, grammar_risk=0.05, usefulness=0.6,
                )
            )
        return out

    def _reason_is_because(self, ctx: RuleContext) -> list[Candidate]:
        """'The reason (why) X is because Y' → 'The reason X is that Y'."""
        out = []
        for m in ctx.finditer(r"is because"):
            sent = next(
                (s for s in ctx.analysis.sentences if s.char_start <= m.start() < s.char_end), None
            )
            if sent is None or "reason" not in ctx.text[sent.char_start : m.start()].lower():
                continue
            original = m.group(0)
            out.append(
                Candidate(
                    rule_id="redundancy.reason_is_because",
                    transform_type=self.transform_type,
                    char_start=m.start(), char_end=m.end(), original=original,
                    replacement=match_case(original, "is that"),
                    reason="'The reason … is because' repeats the idea of cause; 'is that' is cleaner.",
                    confidence=0.8, meaning_risk=0.05, grammar_risk=0.1, usefulness=0.6,
                )
            )
        for m in ctx.finditer(r"the reason why"):
            original = m.group(0)
            out.append(
                Candidate(
                    rule_id="redundancy.reason_why",
                    transform_type=self.transform_type,
                    char_start=m.start(), char_end=m.end(), original=original,
                    replacement=match_case(original, "the reason"),
                    reason="'why' repeats what 'reason' already says.",
                    confidence=0.75, meaning_risk=0.05, grammar_risk=0.1, usefulness=0.4,
                )
            )
        return out
