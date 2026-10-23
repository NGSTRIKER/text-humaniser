"""Fact preservation, logical-consistency and addition/omission checks (§12 A, D, F).

All comparisons are lexical/structural so they are fast and explainable.
A paraphrase that legitimately rewords a fact may be flagged; the policy
treats ambiguous cases conservatively (reject → repair → fall back).
"""

from __future__ import annotations

import re
from collections import Counter

from humanizer.validation.base import CheckResult, Pair, Severity, dispensable_ranges, norm_phrase, related

NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")
NEGATION_CUES = {"not", "n't", "no", "never", "none", "nobody", "nothing", "neither", "nor", "without", "cannot", "nowhere"}
HEDGES = {"might", "may", "could", "possibly", "perhaps", "likely", "unlikely", "probably", "suggest",
          "suggests", "suggested", "appear", "appears", "seem", "seems", "potentially", "apparently", "somewhat"}
STRONG = {"will", "must", "definitely", "certainly", "always", "prove", "proves", "proved", "clearly",
          "undoubtedly", "guarantee", "guarantees", "never"}
CONDITIONS = ("if", "unless", "provided that", "as long as", "only if", "in case")
CAUSAL = ("because", "therefore", "thus", "hence", "consequently", "as a result", "due to",
          "caused", "causes", "cause", "causal", "leads to", "led to", "resulted in", "so that")
COMPARATIVE = ("more", "less", "fewer", "most", "least", "than", "better", "worse")
INTENSIFIERS = {"very", "extremely", "highly", "strong", "strongly", "significant", "significantly",
                "major", "landmark", "huge", "massive", "dramatic", "dramatically", "groundbreaking",
                "crucial", "critical", "key", "vital", "remarkable", "definitive", "completely", "totally"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?|n't", text.lower())


def _count_phrases(text: str, phrases: tuple[str, ...] | set[str]) -> Counter:
    low = " " + re.sub(r"[^a-z' ]+", " ", text.lower()) + " "
    return Counter({p: low.count(f" {p} ") for p in phrases if f" {p} " in low})


def _numbers(text: str) -> Counter:
    return Counter(n.replace(",", "") for n in NUMBER.findall(text))


def extract_comparisons(doc) -> list[tuple[str, str, str]]:
    """Extract (subject_phrase, comparative_lemma, standard_phrase) triples."""
    results = []
    for t in doc.tokens:
        if t.text.lower() != "than" or t.head_index < 0 or t.head_index >= len(doc.tokens):
            continue
        comp_tok = doc.tokens[t.head_index]
        pobj = next((doc.tokens[c] for c in t.children if doc.tokens[c].dep in {"pobj", "dobj"}), None)
        if not pobj and t.children:
            pobj = doc.tokens[t.children[0]]
        if not pobj:
            continue
        gov = (
            doc.tokens[comp_tok.head_index]
            if comp_tok.dep in {"acomp", "dobj", "advmod", "attr", "prep"} and 0 <= comp_tok.head_index < len(doc.tokens)
            else comp_tok
        )
        subj = next((doc.tokens[c] for c in gov.children if "subj" in doc.tokens[c].dep), None)
        if not subj and gov.dep in {"acomp", "attr", "xcomp", "prep"} and 0 <= gov.head_index < len(doc.tokens):
            gov2 = doc.tokens[gov.head_index]
            subj = next((doc.tokens[c] for c in gov2.children if "subj" in doc.tokens[c].dep), None)
        if not subj:
            continue

        def phrase(tok):
            toks = [tok]
            for c in tok.children:
                ct = doc.tokens[c]
                if ct.dep in {"compound", "det", "amod", "poss"}:
                    toks.append(ct)
            toks.sort(key=lambda x: x.index)
            return " ".join(x.text.lower().rstrip(".,!?") for x in toks)

        results.append((phrase(subj), comp_tok.lemma.lower(), phrase(pobj)))
    return results


def check_facts(pair: Pair) -> CheckResult:
    """Entities, numbers/dates/units, negation, modality, conditions, comparisons, causality."""
    o, r = pair.original, pair.rewritten
    res = CheckResult("facts")

    # numbers and quantities (regex is more reliable than NER for raw digits)
    on, rn = _numbers(o.text), _numbers(r.text)
    for n in (on - rn):
        res.add("number", Severity.CRITICAL, f"number {n!r} missing or changed", n, None,
                repair=f"Keep the number {n} exactly as written.")
    for n in (rn - on):
        res.add("number", Severity.CRITICAL, f"number {n!r} appears only in the rewrite", None, n,
                repair=f"Remove the number {n}; it is not in the original.")

    # named entities and dates (normalized, order-insensitive)
    def ents(doc):
        return {norm_phrase(e.text): e for e in doc.entities if not NUMBER.fullmatch(e.text.strip())}
    oe, re_ = ents(o), ents(r)
    rtext = norm_phrase(r.text)
    for key, e in oe.items():
        if key not in re_ and key not in rtext:
            res.add("entity", Severity.CRITICAL, f"{e.label} '{e.text}' missing", e.text,
                    repair=f"Mention '{e.text}' as in the original.")
    otext = norm_phrase(o.text)
    for key, e in re_.items():
        if key not in oe and key not in otext:
            res.add("entity", Severity.CRITICAL, f"new {e.label} '{e.text}' not in original", None, e.text,
                    repair=f"Remove '{e.text}'; it is not in the original.")

    # negation / polarity
    on_neg = sum(_words(o.text).count(c) for c in NEGATION_CUES)
    rn_neg = sum(_words(r.text).count(c) for c in NEGATION_CUES)
    if on_neg != rn_neg:
        res.add("negation", Severity.CRITICAL, f"negation count changed ({on_neg} → {rn_neg})",
                repair="Restore every negation (not, no, never…) exactly where the original has it.")

    # certainty / modality
    ow, rw = set(_words(o.text)), set(_words(r.text))
    lost_hedges = (ow & HEDGES) - rw
    new_strong = (rw & STRONG) - ow
    if lost_hedges and not (rw & HEDGES):
        res.add("certainty", Severity.CRITICAL, f"uncertainty removed ({', '.join(sorted(lost_hedges))})",
                repair=f"Keep the uncertainty of the original ({', '.join(sorted(lost_hedges))}).")
    elif new_strong and ow & HEDGES:
        res.add("certainty", Severity.CRITICAL, f"certainty strengthened ({', '.join(sorted(new_strong))})",
                repair="Do not make tentative statements sound certain.")
    elif new_strong:
        res.add("certainty", Severity.MAJOR, f"stronger wording added ({', '.join(sorted(new_strong))})")

    # conditions
    oc, rc = _count_phrases(o.text, CONDITIONS), _count_phrases(r.text, CONDITIONS)
    if oc != rc:
        sev = Severity.CRITICAL if oc.get("unless", 0) != rc.get("unless", 0) or sum(oc.values()) != sum(rc.values()) else Severity.MAJOR
        res.add("condition", sev, f"conditions changed ({dict(oc)} → {dict(rc)})",
                repair="Keep every condition (if/unless) from the original.")

    # causal claims
    ocs, rcs = _count_phrases(o.text, CAUSAL), _count_phrases(r.text, CAUSAL)
    if sum(rcs.values()) > sum(ocs.values()):
        added = sorted(set(rcs) - set(ocs)) or sorted(rcs)
        res.add("causal", Severity.CRITICAL, f"causal claim added ({', '.join(added)})", None, ", ".join(added),
                repair="Do not add cause-and-effect wording that the original does not state.")
    elif sum(rcs.values()) < sum(ocs.values()):
        res.add("causal", Severity.MAJOR, f"causal wording removed ({', '.join(sorted(set(ocs) - set(rcs)) or ocs)})",
                repair="Keep the cause-and-effect relationship stated in the original.")

    # comparisons
    ocm, rcm = _count_phrases(o.text, COMPARATIVE), _count_phrases(r.text, COMPARATIVE)
    flips = {("more", "less"), ("less", "more"), ("more", "fewer"), ("fewer", "more"), ("better", "worse"), ("worse", "better")}
    if any(a in ocm and b in rcm and b not in ocm for a, b in flips):
        res.add("comparison", Severity.CRITICAL, "comparison direction reversed",
                repair="Keep the direction of every comparison.")
    elif ocm.get("than", 0) != rcm.get("than", 0):
        res.add("comparison", Severity.MAJOR, "comparison added or removed")
    else:
        o_comps = extract_comparisons(o)
        r_comps = extract_comparisons(r)
        for s1, c1, p1 in o_comps:
            for s2, c2, p2 in r_comps:
                if c1 == c2 and (s1 == p2 or s1 in p2 or p2 in s1) and (p1 == s2 or p1 in s2 or s2 in p1) and s1 != p1:
                    res.add("comparison", Severity.CRITICAL, f"comparison direction reversed ('{s1}' vs '{p1}')",
                            f"{s1} ... {p1}", f"{s2} ... {p2}",
                            repair=f"Keep the original comparison direction: '{s1}' was compared against '{p1}'.")
    return res


def check_additions_omissions(pair: Pair) -> CheckResult:
    """Unsupported adjectives/intensifiers/details and lost content words (§12 F)."""
    o, r = pair.original, pair.rewritten
    res = CheckResult("additions_omissions")
    o_lemmas = {t.lemma.lower() for t in o.tokens}
    o_content = {t.lemma.lower(): t.pos for t in o.tokens if t.pos in {"NOUN", "PROPN", "ADJ", "VERB", "ADV"} and not t.is_stop}

    def supported(lemma: str, pos: str) -> bool:
        if lemma in o_lemmas:
            return True
        return any(related(lemma, ol, None) for ol, op in o_content.items() if op == pos or pos in {"ADJ", "ADV"})

    added = []
    for t in r.tokens:
        lem = t.lemma.lower()
        if t.is_punct or t.is_stop or lem in o_lemmas:
            continue
        if lem in INTENSIFIERS or (t.pos in {"ADJ", "ADV"} and not supported(lem, t.pos)) or (
            t.pos == "NOUN" and t.dep in {"compound", "amod"} and not supported(lem, t.pos)
        ):
            added.append(t.text)
    for w in dict.fromkeys(added):
        sev = Severity.MAJOR if w.lower() in INTENSIFIERS else Severity.MINOR
        res.add("addition", sev, f"unsupported modifier '{w}'", None, w,
                repair=f"Remove '{w}'; the original does not say this.")

    # omissions: content nouns/verbs from the original with no equivalent in the rewrite
    r_lemmas = {t.lemma.lower() for t in r.tokens}
    r_content = [(t.lemma.lower(), t.pos) for t in r.tokens if t.pos in {"NOUN", "PROPN", "VERB", "ADJ"}]
    skip = dispensable_ranges(o.text)
    dispensable = {t.lemma.lower() for t in o.tokens if any(a <= t.char_start < b for a, b in skip)}
    important = [(lem, pos) for lem, pos in o_content.items()
                 if pos in {"NOUN", "PROPN", "VERB"} and lem not in {"be", "have", "do"} and lem not in dispensable]
    lost = [lem for lem, pos in important
            if lem not in r_lemmas and not any(related(lem, rl, None) for rl, rp in r_content if rp == pos)]
    if important:
        res.score = round(1 - len(lost) / len(important), 3)
    for lem in lost:
        res.add("omission", Severity.MINOR, f"'{lem}' has no counterpart in the rewrite", lem)
    if important and len(lost) / len(important) > 0.3 and len(lost) >= 2:
        res.add("omission", Severity.CRITICAL, f"{len(lost)} of {len(important)} key words lost",
                repair="Keep all the information from the original: " + ", ".join(lost[:8]) + ".")
    return res
