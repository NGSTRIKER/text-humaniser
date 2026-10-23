"""Contextual sense selection and fit scoring for Stage 4.

* :func:`disambiguate` — a simplified Lesk algorithm: the sense whose
  definition/examples/related words overlap most with the surrounding
  sentence wins, with WordNet's frequency order as a tie-breaking prior.
* :class:`EmbeddingScorer` — OPTIONAL.  Uses spaCy word vectors when a
  model that ships them (``en_core_web_md``/``lg``) is installed.  The
  small default model has no real vectors, so this simply switches off.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

from humanizer.analyzer import DocumentAnalysis
from humanizer.logging_config import get_logger
from humanizer.word_graph import SenseNode, WordGraph

log = get_logger("context_analyzer")

_STOP = frozenset(
    "a an the of to in on at by for with from and or but is are was were be been being "
    "it its this that these those as which who whom what when where how than then so such "
    "not no or into out up down over under some any all each one other used especially "
    "something someone".split()
)


def _words(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _STOP or len(w) < 3:
            continue
        out.add(w.rstrip("s") if len(w) > 4 else w)  # crude stemming
    return out


@dataclass(frozen=True)
class SenseGuess:
    sense: SenseNode
    score: float
    confidence: float   # 0–1: how clearly this sense beats the others


def context_words(analysis: DocumentAnalysis, sentence_index: int, window: int = 1) -> set[str]:
    """Content lemmas from the sentence and ``window`` neighbours each side."""
    lo, hi = max(0, sentence_index - window), sentence_index + window
    out: set[str] = set()
    for tok in analysis.tokens:
        if lo <= tok.sentence_index <= hi and not tok.is_stop and not tok.is_punct and tok.pos in {
            "NOUN", "VERB", "ADJ", "ADV", "PROPN"
        }:
            out |= _words(tok.lemma)
    return out


def disambiguate(
    graph: WordGraph, word: str, wn_pos: str, context: set[str], exclude: str = ""
) -> list[SenseGuess]:
    """Rank the senses of ``word`` for this context (best first)."""
    senses = graph.senses(word, wn_pos)
    if not senses:
        return []
    ctx = context - _words(exclude or word)
    raw: list[tuple[SenseNode, float]] = []
    for s in senses:
        signature = _words(s.definition) | _words(" ".join(s.examples))
        signature |= _words(" ".join(s.lemma_names)) | _words(" ".join(graph.hypernym_words(s.sense_id)))
        overlap = len(signature & ctx)
        prior = 1.0 / (s.rank + 1)               # WordNet lists frequent senses first
        usage = math.log1p(graph.lemma_count(word, s.sense_id)) * 0.3
        raw.append((s, overlap + prior + usage))
    raw.sort(key=lambda x: x[1], reverse=True)
    if len(raw) == 1:
        return [SenseGuess(raw[0][0], raw[0][1], 1.0)]
    # confidence = softmax probability of each sense
    top = raw[0][1]
    exps = [math.exp(sc - top) for _, sc in raw]
    total = sum(exps)
    return [SenseGuess(s, sc, e / total) for (s, sc), e in zip(raw, exps)]


class EmbeddingScorer:
    """Optional vector similarity; ``available`` is False without real vectors."""

    def __init__(self, model: str = "en_core_web_md") -> None:
        self.nlp: Any = None
        try:
            import spacy

            nlp = spacy.load(model)
            if nlp.vocab.vectors.shape[0] > 0:
                self.nlp = nlp
                log.info("Embedding scorer using %s vectors", model)
        except Exception:  # missing model or spaCy → silently disabled
            log.info("Embedding model %s not available; using WordNet-only ranking", model)

    @property
    def available(self) -> bool:
        return self.nlp is not None

    def similarity(self, a: str, b: str) -> Optional[float]:
        if not self.available:
            return None
        va, vb = self.nlp.vocab[a.lower()], self.nlp.vocab[b.lower()]
        if not (va.has_vector and vb.has_vector):
            return None
        return float(va.similarity(vb))
