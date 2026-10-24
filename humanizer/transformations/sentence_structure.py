"""Sentence combining, splitting and (opt-in) reordering (Stage 5).

Combining only uses "and" for two short sentences with the SAME subject:
    "She opened the door. She walked in." → "She opened the door and walked in."
It never invents "because", "therefore" or "although".

Splitting breaks a long sentence at "; " or ", and" between two complete
clauses (each with its own subject).

Reordering swaps two adjacent, logically independent body sentences.  It
is OFF by default (``sentence_reordering=False``).
"""

from __future__ import annotations

from typing import Any, Optional

from humanizer.candidates import Candidate
from humanizer.config import Intensity
from humanizer.paragraph_flow import analyze_flow, swappable_pairs, swappable_paragraphs
from humanizer.transformations.base import Rule, RuleContext
from humanizer.transformations.syntax_utils import capitalize_first, parse, span_text, tidy

MAX_COMBINE_WORDS = 18
MIN_SPLIT_WORDS = 18


def _clause(doc: Any) -> Optional[tuple[Any, Any]]:
    """(root verb, subject) for a single simple declarative clause."""
    sents = list(doc.sents)
    if len(sents) != 1:
        return None
    root = sents[0].root
    if root.pos_ not in {"VERB", "AUX"}:
        return None
    subj = next((c for c in root.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
    if subj is None:
        return None
    return root, subj


def _agrees(noun: Any, pronoun: str) -> bool:
    """Does ``pronoun`` plausibly refer to the noun-phrase head ``noun``?"""
    plural = noun.tag_ in {"NNS", "NNPS"} or any(c.dep_ == "conj" for c in noun.children)
    person = noun.ent_type_ == "PERSON" or (noun.pos_ == "PROPN" and not noun.ent_type_)
    if pronoun == "they":
        return plural
    if pronoun in {"he", "she"}:
        return person and not plural
    if pronoun == "it":
        return not plural and not person
    return False


def combine(first: str, second: str, ambiguous_pronoun: bool = False) -> Optional[str]:
    """Join two sentences with the same subject using 'and'."""
    if not (first.rstrip().endswith(".") and second.rstrip().endswith(".")):
        return None
    if len(first.split()) > MAX_COMBINE_WORDS or len(second.split()) > MAX_COMBINE_WORDS:
        return None
    d1, d2 = parse(first.strip()), parse(second.strip())
    c1, c2 = _clause(d1), _clause(d2)
    if not c1 or not c2:
        return None
    (r1, s1), (r2, s2) = c1, c2
    if any(c.dep_ in {"conj", "cc", "advcl", "ccomp"} for c in r1.children):
        return None
    if any(c.dep_ in {"conj", "cc"} for c in r2.children):
        return None
    s2_toks = list(s2.subtree)
    if s2_toks[0].i != 0:
        return None  # something (marker, phrase) precedes the subject
    s1_text = span_text(s1.subtree).lower()
    s2_text = span_text(s2_toks).lower()
    same = s1_text == s2_text
    pronoun_ok = (
        len(s2_toks) == 1 and s2.pos_ == "PRON" and not ambiguous_pronoun
        and s1.pos_ != "PRON" and len(list(s1.subtree)) <= 4
        and _agrees(s1, s2.lower_)
    )
    if not (same or pronoun_ok):
        return None
    rest = [t for t in d2 if t.i > s2_toks[-1].i]
    head = first.rstrip()[:-1]
    return tidy(f"{head} and {span_text(rest)}")


def split(sentence: str) -> Optional[tuple[str, str]]:
    """Split a long two-clause sentence. Returns (new_text, how) or None."""
    text = sentence.strip()
    if len(text.split()) < MIN_SPLIT_WORDS or not text.endswith("."):
        return None
    if text.count("; ") == 1:
        a, b = text.split("; ")
        if _clause(parse(a + ".")) and _clause(parse(b)):
            return tidy(f"{a}. {capitalize_first(b)}"), "semicolon"
    doc = parse(text)
    for tok in doc:
        if tok.lower_ in {"and", "but"} and tok.dep_ == "cc" and tok.i > 0 and doc[tok.i - 1].text == ",":
            head = tok.head
            conj = next((c for c in head.children if c.dep_ == "conj" and c.i > tok.i), None)
            if head.dep_ != "ROOT" or conj is None or conj.pos_ not in {"VERB", "AUX"}:
                continue
            if not any(c.dep_ in {"nsubj", "nsubjpass", "expl"} for c in conj.children):
                continue
            a = span_text(doc[: tok.i - 1])
            b = span_text(doc[tok.i + 1 :])
            if len(a.split()) < 5 or len(b.split()) < 5:
                continue
            return tidy(f"{a}. {capitalize_first(b)}"), f"comma-{tok.lower_}"
    return None


class SentenceStructureRule(Rule):
    rule_id = "sentence_structure"
    description = "Combine short same-subject sentences or split long two-clause sentences."
    transform_type = "sentence_structure"
    min_intensity = Intensity.HEAVY

    def __init__(self, reordering: bool = False) -> None:
        self.reordering = reordering

    def propose(self, ctx: RuleContext) -> list[Candidate]:
        a = ctx.analysis
        sents = a.sentences
        flow = analyze_flow(a)
        out: list[Candidate] = []

        for i in range(len(sents) - 1):
            s1, s2 = sents[i], sents[i + 1]
            if flow.sentences[i].paragraph != flow.sentences[i + 1].paragraph:
                continue
            gap = ctx.text[s1.char_end : s2.char_start]
            if "\n" in gap:
                continue
            ambiguous = any(p.sentence_index == i + 1 and p.ambiguous for p in a.pronouns)
            joined = combine(s1.text, s2.text, ambiguous)
            if joined:
                out.append(self._cand(
                    s1.char_start, s2.char_end, ctx.text[s1.char_start : s2.char_end], joined,
                    "sentence_structure.combine", 0.12,
                    "Combined two short sentences with the same subject using 'and' (no new logical connector).",
                ))

        for s in sents:
            res = split(s.text)
            if res:
                out.append(self._cand(
                    s.char_start, s.char_end, s.text, res[0], f"sentence_structure.split_{res[1].replace('-', '_')}",
                    0.1, "Split a long sentence into two complete sentences.",
                ))

        if self.reordering:
            for x, y in swappable_pairs(flow):
                s1, s2 = sents[x], sents[y]
                gap = ctx.text[s1.char_end : s2.char_start]
                out.append(self._cand(
                    s1.char_start, s2.char_end, ctx.text[s1.char_start : s2.char_end],
                    s2.text + gap + s1.text, "paragraph_flow.swap", 0.3,
                    "Reordered two logically independent sentences within the paragraph.",
                    transform="paragraph_flow",
                ))
            for p1, p2 in swappable_paragraphs(flow, a):
                p1_start = sents[flow.paragraphs[p1][0]].char_start
                p1_end = sents[flow.paragraphs[p1][-1]].char_end
                p2_start = sents[flow.paragraphs[p2][0]].char_start
                p2_end = sents[flow.paragraphs[p2][-1]].char_end
                gap = ctx.text[p1_end:p2_start]
                p1_text = ctx.text[p1_start:p1_end]
                p2_text = ctx.text[p2_start:p2_end]
                out.append(self._cand(
                    p1_start, p2_end, ctx.text[p1_start:p2_end],
                    p2_text + gap + p1_text, "paragraph_flow.paragraph_swap", 0.25,
                    "Reordered two logically independent paragraphs to improve document flow.",
                    transform="paragraph_flow",
                ))
        return out

    def _cand(self, start: int, end: int, original: str, new: str, rule_id: str,
              risk: float, reason: str, transform: str | None = None) -> Candidate:
        return Candidate(
            rule_id=rule_id, transform_type=transform or self.transform_type,
            char_start=start, char_end=end, original=original, replacement=new,
            reason=reason, confidence=0.7, meaning_risk=risk, grammar_risk=0.12,
            usefulness=0.55, context_fit=0.7, diversity_gain=0.6, preserves_protected=True,
            priority=2,
        )
