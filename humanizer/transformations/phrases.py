"""Safe phrase transformations: weak verb + noun → strong verb.

"make a decision" → "decide", keeping tense: "made a decision" → "decided",
"has come to a conclusion" → "has concluded".
"""

from __future__ import annotations

from dataclasses import dataclass

from humanizer.candidates import Candidate
from humanizer.config import Intensity
from humanizer.transformations.base import Rule, RuleContext, match_case


@dataclass(frozen=True)
class Nominalization:
    key: str
    forms: tuple[str, str, str, str, str]    # base, 3sg, past, -ing, past participle
    tail: str                                # regex after the verb
    target: tuple[str, str, str, str]        # base, 3sg, past(=pp), -ing
    meaning_risk: float = 0.1


ITEMS: tuple[Nominalization, ...] = (
    Nominalization("make_a_decision", ("make", "makes", "made", "making", "made"), r"a decision", ("decide", "decides", "decided", "deciding")),
    Nominalization("come_to_a_conclusion", ("come", "comes", "came", "coming", "come"), r"to a conclusion", ("conclude", "concludes", "concluded", "concluding")),
    Nominalization("give_consideration_to", ("give", "gives", "gave", "giving", "given"), r"(?:careful )?consideration to", ("consider", "considers", "considered", "considering")),
    Nominalization("take_into_consideration", ("take", "takes", "took", "taking", "taken"), r"into consideration", ("consider", "considers", "considered", "considering"), 0.15),
    Nominalization("conduct_an_investigation", ("conduct", "conducts", "conducted", "conducting", "conducted"), r"an investigation (?:into|of)", ("investigate", "investigates", "investigated", "investigating")),
    Nominalization("perform_an_analysis", ("perform", "performs", "performed", "performing", "performed"), r"an analysis of", ("analyze", "analyzes", "analyzed", "analyzing")),
    Nominalization("provide_assistance", ("provide", "provides", "provided", "providing", "provided"), r"assistance to", ("help", "helps", "helped", "helping"), 0.12),
    Nominalization("make_an_attempt", ("make", "makes", "made", "making", "made"), r"an attempt to", ("try to", "tries to", "tried to", "trying to")),
    Nominalization("reach_an_agreement", ("reach", "reaches", "reached", "reaching", "reached"), r"an agreement", ("agree", "agrees", "agreed", "agreeing"), 0.12),
)


class NominalizationRule(Rule):
    rule_id = "phrase"
    description = "Replace a weak verb + abstract noun with the matching verb."
    transform_type = "phrase"
    min_intensity = Intensity.MODERATE

    def propose(self, ctx: RuleContext) -> list[Candidate]:
        out: list[Candidate] = []
        for item in ITEMS:
            verbs = "|".join(sorted(set(item.forms), key=len, reverse=True))
            for m in ctx.finditer(rf"({verbs}) {item.tail}"):
                original = m.group(0)
                verb = m.group(1).lower()
                target = self._inflect(ctx, m.start(), verb, item)
                note = "" if target else "could not determine the verb form"
                out.append(
                    Candidate(
                        rule_id=f"phrase.{item.key}", transform_type=self.transform_type,
                        char_start=m.start(), char_end=m.end(), original=original,
                        replacement=match_case(original, target) if target else None,
                        reason=f"Restructured phrase: {original!r} → {target!r} (same action, fewer words).",
                        confidence=0.75, meaning_risk=item.meaning_risk, grammar_risk=0.15,
                        usefulness=0.6, context_fit=0.7, grammar_ok=bool(target), note=note,
                    )
                )
        return out

    @staticmethod
    def _inflect(ctx: RuleContext, start: int, verb: str, item: Nominalization) -> str:
        base, s3, past, ing, pp = item.forms
        prev = ctx.token_before(start)
        after_have = prev is not None and prev.lemma.lower() == "have"
        if verb == pp and after_have:
            return item.target[2]
        if verb == base:
            return item.target[0]
        if verb == s3:
            return item.target[1]
        if verb == past:
            return item.target[2]
        if verb == ing:
            return item.target[3]
        return ""
