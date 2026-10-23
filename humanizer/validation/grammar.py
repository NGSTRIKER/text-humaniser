"""Grammatical and stylistic validation of the rewrite (§12 E).

The rewrite is re-analyzed and only problems the rewrite *introduced*
(not already present in the original) are reported.
"""

from __future__ import annotations

import re
from collections import Counter

from humanizer.analyzer import DocumentAnalysis
from humanizer.validation.base import CheckResult, Pair, Severity

RUN_ON_WORDS = 55


def _duplicates(text: str) -> set[str]:
    return {m.group(1).lower() for m in re.finditer(r"\b(\w+)\s+\1\b", text, re.IGNORECASE)
            if not m.group(1).isdigit() and m.group(1).lower() not in {"that", "had"}}


def _fragments(doc: DocumentAnalysis) -> list[str]:
    out = []
    for s in doc.sentences:
        toks = [doc.tokens[i] for i in s.token_indices if not doc.tokens[i].is_punct]
        if len(toks) >= 3 and not any(t.pos in {"VERB", "AUX"} for t in toks):
            out.append(s.text)
    return out


def _agreement_errors(doc: DocumentAnalysis) -> list[str]:
    errs = []
    for t in doc.tokens:
        if t.dep != "nsubj" or t.head_index < 0:
            continue
        v = doc.tokens[t.head_index]
        if t.tag in {"NNS", "NNPS"} and v.tag == "VBZ" and t.lemma.lower() not in {"news", "series"}:
            errs.append(f"{t.text} {v.text}")
        elif t.tag in {"NN", "NNP"} and v.tag == "VBP":
            if not any(doc.tokens[c].dep == "conj" for c in t.children):
                errs.append(f"{t.text} {v.text}")
    return errs


def _trigrams(doc: DocumentAnalysis) -> Counter:
    words = [t.lower_ if hasattr(t, "lower_") else t.text.lower() for t in doc.tokens if not t.is_punct]
    return Counter(" ".join(words[i:i + 3]) for i in range(len(words) - 2))


def _main_tenses(doc: DocumentAnalysis) -> Counter:
    return Counter(c.tense for s in doc.sentences for c in s.clauses if c.clause_type == "main" and c.tense)


def check_grammar(pair: Pair) -> CheckResult:
    o, r = pair.original, pair.rewritten
    res = CheckResult("grammar")

    for w in _duplicates(r.text) - _duplicates(o.text):
        res.add("grammar", Severity.MAJOR, f"duplicated word '{w} {w}'", None, w, repair=f"Remove the duplicated '{w}'.")
    for frag in set(_fragments(r)) - set(_fragments(o)):
        res.add("grammar", Severity.MAJOR, "sentence fragment (no verb)", None, frag,
                repair="Make every sentence complete.")
    for e in set(_agreement_errors(r)) - set(_agreement_errors(o)):
        res.add("grammar", Severity.MAJOR, f"possible subject-verb agreement error: '{e}'", None, e,
                repair=f"Fix agreement in '{e}'.")
    o_long = sum(len(s.text.split()) > RUN_ON_WORDS for s in o.sentences)
    if sum(len(s.text.split()) > RUN_ON_WORDS for s in r.sentences) > o_long:
        res.add("style", Severity.MINOR, f"run-on sentence (over {RUN_ON_WORDS} words)")

    ot, rt = _trigrams(o), _trigrams(r)
    for g, n in rt.items():
        if n > 1 and n > ot.get(g, 0) and not all(w in {"of", "the", "a", "to", "and", "in"} for w in g.split()):
            res.add("style", Severity.MINOR, f"repeated phrase introduced: '{g}'", None, g)

    if sum(p.ambiguous for p in r.pronouns) > sum(p.ambiguous for p in o.pronouns):
        res.add("style", Severity.MINOR, "new ambiguous pronoun", repair="Replace unclear pronouns with the noun.")

    ot_, rt_ = _main_tenses(o), _main_tenses(r)
    if ot_ and rt_ and ot_.most_common(1)[0][0] != rt_.most_common(1)[0][0]:
        res.add("tense", Severity.MAJOR, f"dominant tense changed ({ot_.most_common(1)[0][0]} → {rt_.most_common(1)[0][0]})",
                repair="Keep the original tense.")
    if len(r.sentences) and len(o.sentences) and len(r.sentences) > 2 * len(o.sentences) + 1:
        res.add("style", Severity.MINOR, "text heavily fragmented into many sentences")
    return res
