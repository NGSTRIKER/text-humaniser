"""Semantic similarity signal (§12 C) — advisory only.

Uses spaCy document vectors when an md/lg model is loaded (``use_embeddings``),
otherwise a lemma-overlap score.  A low score only produces a MAJOR issue
when the text was not heavily restructured; it never rejects on its own.
"""

from __future__ import annotations

from typing import Any, Optional

from humanizer.validation.base import CheckResult, Pair, Severity


def lexical_overlap(pair: Pair) -> float:
    from humanizer.validation.base import related

    def bag(doc):
        return {t.lemma.lower() for t in doc.tokens if not t.is_punct and not t.is_stop}
    a, b = bag(pair.original), bag(pair.rewritten)
    if not a and not b:
        return 1.0
    matched_b: set[str] = set()
    matches = 0
    for wa in a:
        if wa in b:
            matches += 1
            matched_b.add(wa)
        elif any(related(wa, wb, None) for wb in b if wb not in matched_b):
            matches += 1
            for wb in b:
                if wb not in matched_b and related(wa, wb, None):
                    matched_b.add(wb)
                    break
    total = len(a | b)
    return matches / total if total > 0 else 1.0


def embedding_similarity(original: str, rewritten: str, nlp: Any) -> Optional[float]:
    try:
        d1, d2 = nlp(original), nlp(rewritten)
        if not (d1.has_vector and d2.has_vector) or not d1.vector_norm or not d2.vector_norm:
            return None
        return float(d1.similarity(d2))
    except Exception:
        return None


def check_similarity(pair: Pair, threshold: float, vector_nlp: Any = None) -> CheckResult:
    res = CheckResult("similarity")
    emb = embedding_similarity(pair.original.text, pair.rewritten.text, vector_nlp) if vector_nlp else None
    if emb is not None:
        res.score, res.note = round(emb, 3), "embedding"
        low = emb < threshold
    else:
        res.score, res.note = round(lexical_overlap(pair), 3), "lexical overlap (no vectors)"
        low = res.score < threshold * 0.4  # lexical overlap is naturally much lower
    if low:
        res.add("similarity", Severity.MAJOR, f"low similarity {res.score:.2f} ({res.note})",
                repair="Stay closer to the original content.")
    return res
