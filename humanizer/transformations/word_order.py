"""Phrase rearrangement (Stage 5).

* Manner adverb repositioning:
  "She carefully opened the door." ⇄ "She opened the door carefully."
* Introductory phrase movement:
  "After lunch, we met the client." → "We met the client after lunch."
  and the reverse when it breaks a run of identical sentence openings.

Scope-sensitive adverbs ("only", "even", "just", "also", "never" …) are
never moved, because moving them changes what they apply to.
"""

from __future__ import annotations

from typing import Any, Optional

from humanizer.candidates import Candidate
from humanizer.config import Intensity
from humanizer.transformations.base import Rule, RuleContext
from humanizer.transformations.syntax_utils import (
    capitalize_first,
    decapitalize,
    final_punct,
    parse,
    span_text,
    subtree,
    tidy,
)

SCOPE_ADVERBS = frozenset(
    "only even just also too either merely simply really actually never always often "
    "sometimes usually still already almost nearly hardly barely rarely seldom not "
    "probably possibly certainly definitely perhaps maybe largely mostly mainly".split()
)
SAFE_CLAUSE_MARKS = frozenset({"when", "after", "before", "while", "once", "if", "until"})


def _root(doc: Any) -> Optional[Any]:
    sents = list(doc.sents)
    if len(sents) != 1:
        return None
    root = sents[0].root
    return root if root.pos_ in {"VERB", "AUX"} else None


def move_adverb(sentence: str) -> Optional[tuple[str, str]]:
    """Move a manner adverb between pre-verb and sentence-final position.

    Returns ``(new_sentence, direction)`` or ``None``.
    """
    doc = parse(sentence.strip())
    root = _root(doc)
    punct = final_punct(doc)
    if root is None or punct != "." or sentence.strip().endswith("?"):
        return None
    kids = list(root.children)
    obj = next((c for c in kids if c.dep_ in {"dobj", "obj"}), None)
    if obj is None or any(c.dep_ == "neg" for c in kids):
        return None
    words = [t for t in doc if not t.is_punct]
    if not words:
        return None
    last = words[-1]
    for adv in kids:
        if adv.dep_ != "advmod" or adv.tag_ != "RB" or not adv.lower_.endswith("ly"):
            continue
        if adv.lower_ in SCOPE_ADVERBS or len(list(adv.children)):
            continue
        # pre-verb → end: "She carefully opened the door."
        if adv.i == root.i - 1 and obj.right_edge.i == last.i:
            rest = [t for t in doc if t.i != adv.i and not (t.is_punct and t.i == len(doc) - 1)]
            text = span_text(rest)
            if adv.i == 0:
                text = capitalize_first(text)
            return tidy(f"{text} {adv.lower_}{punct}"), "adverb to end"
        # end → pre-verb: "She opened the door carefully."
        if adv.i == last.i and adv.i - 1 == obj.right_edge.i:
            before = [t for t in doc if t.i < root.i]
            after = [t for t in doc if root.i <= t.i < adv.i]
            text = f"{span_text(before)} {adv.lower_} {span_text(after)}{punct}"
            return tidy(text), "adverb before verb"
    return None


def front_to_back(sentence: str) -> Optional[str]:
    """'After lunch, we met the client.' → 'We met the client after lunch.'"""
    doc = parse(sentence.strip())
    root = _root(doc)
    punct = final_punct(doc)
    if root is None or punct != ".":
        return None
    comma = next((t for t in doc if t.text == ","), None)
    if comma is None or comma.i > 6:
        return None
    intro = next((c for c in root.children if c.i < comma.i and c.right_edge.i == comma.i - 1 and c.left_edge.i == 0), None)
    if intro is None:
        return None
    if intro.dep_ == "advcl":
        mark = next((c for c in intro.children if c.dep_ == "mark"), None)
        if mark is None or mark.lower_ not in SAFE_CLAUSE_MARKS:
            return None
    elif intro.dep_ not in {"prep", "npadvmod"}:
        return None
    main = [t for t in doc if t.i > comma.i and t.i != len(doc) - 1]
    if len(main) > 15 or any(t.text == "," for t in main):
        return None
    intro_toks = subtree(intro)
    intro_text = decapitalize(intro_toks, span_text(intro_toks))
    main_text = capitalize_first(span_text(main))
    return tidy(f"{main_text} {intro_text}{punct}")


def back_to_front(sentence: str) -> Optional[str]:
    """Move a final time/condition phrase to the front: 'We met after lunch.' → 'After lunch, we met.'"""
    doc = parse(sentence.strip())
    root = _root(doc)
    punct = final_punct(doc)
    if root is None or punct != "." or any(t.text == "," for t in doc):
        return None
    words = [t for t in doc if not t.is_punct]
    if not words:
        return None
    last = words[-1]
    tail = next((c for c in root.children if c.i > root.i and c.right_edge.i == last.i), None)
    if tail is None:
        return None
    toks = subtree(tail)
    if tail.dep_ == "prep":
        if tail.lower_ not in {"after", "before", "during", "in", "on", "at", "since"}:
            return None
        if not any(t.ent_type_ in {"DATE", "TIME"} for t in toks) and tail.lower_ not in {"after", "before", "during"}:
            return None
    elif tail.dep_ == "advcl":
        mark = next((c for c in tail.children if c.dep_ == "mark"), None)
        if mark is None or mark.lower_ not in SAFE_CLAUSE_MARKS:
            return None
    else:
        return None
    main = [t for t in doc if t.i < toks[0].i]
    return tidy(f"{capitalize_first(span_text(toks))}, {decapitalize(main, span_text(main))}{punct}")


class PhraseOrderRule(Rule):
    rule_id = "word_order"
    description = "Reposition adverbs and introductory phrases without changing scope."
    transform_type = "word_order"
    min_intensity = Intensity.MODERATE

    def propose(self, ctx: RuleContext) -> list[Candidate]:
        out: list[Candidate] = []
        sents = ctx.analysis.sentences
        for i, sent in enumerate(sents):
            moved = move_adverb(sent.text)
            if moved:
                out.append(self._cand(sent, moved[0], f"word_order.{moved[1].replace(' ', '_')}", 0.08,
                                      f"Rearranged a phrase: moved the adverb ({moved[1]}).", 0.35))
            back = front_to_back(sent.text)
            if back:
                out.append(self._cand(sent, back, "word_order.intro_to_end", 0.12,
                                      "Rearranged a phrase: moved the introductory phrase to the end.", 0.35))
            repeated = i > 0 and sent.opening and sent.opening.lower() == sents[i - 1].opening.lower()
            if repeated:
                front = back_to_front(sent.text)
                if front:
                    out.append(self._cand(sent, front, "word_order.end_to_front", 0.1,
                                          "Changed the sentence opening to avoid repeating the previous one.", 0.6))
        return out

    def _cand(self, sent: Any, new: str, rule_id: str, risk: float, reason: str, use: float) -> Candidate:
        return Candidate(
            rule_id=rule_id, transform_type=self.transform_type,
            char_start=sent.char_start, char_end=sent.char_end, original=sent.text,
            replacement=new, reason=reason, confidence=0.75, meaning_risk=risk,
            grammar_risk=0.1, usefulness=use, context_fit=0.7, diversity_gain=0.4,
            preserves_protected=True,
        )
