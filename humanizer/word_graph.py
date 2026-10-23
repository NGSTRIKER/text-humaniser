"""Stage 4 — Word relationship graph backed by WordNet.

Nodes are *word senses* (WordNet synsets), so one spelling with several
meanings ("bank" = river side vs. money) maps to several nodes.  Edges
carry a typed :class:`Relation`; synonyms, near-synonyms, antonyms and
broader/narrower terms are deliberately NOT interchangeable.

The graph only *offers* alternatives.  Deciding whether one fits the
sentence is the job of :mod:`humanizer.context_analyzer` and the
synonym rule.

Requires NLTK's WordNet data:
    uv run python -m nltk.downloader wordnet omw-1.4
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Iterable, Optional

from humanizer.logging_config import get_logger

log = get_logger("word_graph")


class WordGraphUnavailableError(RuntimeError):
    """Raised when NLTK or its WordNet data is missing."""


class Relation(str, Enum):
    SYNONYM = "synonym"          # same sense (same synset)
    SIMILAR = "similar"          # near-synonym (adjective "similar to" / "also see")
    ANTONYM = "antonym"          # opposite — never a substitution candidate
    BROADER = "broader"          # hypernym
    NARROWER = "narrower"        # hyponym
    DERIVED = "derived"          # morphological relative ("decide" ↔ "decision")


# spaCy coarse POS → WordNet POS
POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

# Small curated register lists.  Neutral is the default.
INFORMAL = frozenset({
    "kid", "kids", "guy", "guys", "stuff", "lots", "awesome", "cool", "okay", "ok",
    "nab", "grab", "flick", "booze", "cop", "buck", "bucks", "gonna", "wanna", "kinda",
    "freak", "nuts", "crap", "dumb", "boss", "chuck", "snag", "pricey", "loads",
    "bash", "bang", "bug", "fella", "gal", "folks", "nope", "yeah", "hassle",
    # colloquial / informal-group words: fine in casual text, wrong elsewhere
    "squad", "crew", "gang", "buddy", "buddies", "pal", "pals", "mate", "mates", "chap",
    "bloke", "dude", "gig", "kinda", "sorta", "flop", "shot", "whopping", "hefty", "ditch",
})
# Measured formal words: fine (slightly preferred) in academic/professional text.
ELEVATED = frozenset({
    "demonstrate", "indicate", "substantial", "considerable", "examine", "assess", "evaluate",
    "obtain", "require", "sufficient", "approximately", "consequently", "therefore", "however",
    "significant", "furthermore", "moreover", "establish", "illustrate", "investigate",
    "determine", "maintain", "primary", "numerous", "additional", "particular", "propose",
    "objective", "approach", "method", "outcome", "conclude", "suggest", "observe", "identify",
})
FORMAL = frozenset({
    "commence", "terminate", "utilize", "endeavor", "ascertain", "facilitate",
    "whilst", "thereby", "henceforth", "aforementioned", "pursuant", "remuneration",
    "procure", "promulgate", "effectuate", "heretofore", "notwithstanding",
    "menage", "mortal", "betwixt", "hitherto", "whence",
})


def register_of(word: str) -> str:
    w = word.lower()
    if w in INFORMAL:
        return "informal"
    if w in FORMAL:
        return "formal"  # ornate: avoided in every style unless heavy + neutral
    if w in ELEVATED:
        return "elevated"
    return "neutral"


@dataclass(frozen=True)
class SenseNode:
    """One meaning of a word."""

    sense_id: str                 # e.g. "result.n.03"
    pos: str                      # WordNet pos: n v a r (s→a)
    definition: str
    examples: tuple[str, ...]
    lemma_names: tuple[str, ...]  # words that share this sense
    rank: int = 0                 # position in WordNet's list (≈ frequency order)


@dataclass(frozen=True)
class Edge:
    """A typed relationship between two words in a given sense."""

    source: str
    target: str
    relation: Relation
    sense_id: str
    target_sense_id: str
    weight: float = 1.0           # usage count in that sense (SemCor), +1 smoothing


@lru_cache(maxsize=1)
def _wordnet() -> Any:
    try:
        from nltk.corpus import wordnet as wn
    except ImportError as exc:  # pragma: no cover
        raise WordGraphUnavailableError("NLTK is not installed. Run: uv add nltk") from exc
    try:
        wn.ensure_loaded()
    except LookupError as exc:
        raise WordGraphUnavailableError(
            "WordNet data missing. Run: uv run python -m nltk.downloader wordnet omw-1.4"
        ) from exc
    return wn


def is_available() -> bool:
    try:
        _wordnet()
        return True
    except WordGraphUnavailableError:
        return False


def _norm_pos(pos: str) -> str:
    return "a" if pos == "s" else pos


def _word(lemma_name: str) -> str:
    return lemma_name.replace("_", " ")


class WordGraph:
    """Lazy, cached view of WordNet as a typed graph."""

    def __init__(self) -> None:
        self.wn = _wordnet()

    # ── nodes ────────────────────────────────────────────────────
    @lru_cache(maxsize=4096)
    def senses(self, word: str, pos: Optional[str] = None) -> tuple[SenseNode, ...]:
        """All senses of ``word`` (optionally restricted to a WordNet POS)."""
        out = []
        for rank, syn in enumerate(self.wn.synsets(word.lower(), pos=pos)):
            if pos == "a" and syn.pos() not in {"a", "s"}:
                continue
            out.append(
                SenseNode(
                    sense_id=syn.name(), pos=_norm_pos(syn.pos()),
                    definition=syn.definition(), examples=tuple(syn.examples()),
                    lemma_names=tuple(_word(l.name()) for l in syn.lemmas()), rank=rank,
                )
            )
        return tuple(out)

    def hypernym_words(self, sense_id: str) -> tuple[str, ...]:
        syn = self.wn.synset(sense_id)
        return tuple(_word(l.name()) for h in syn.hypernyms() for l in h.lemmas())

    # ── edges ────────────────────────────────────────────────────
    @lru_cache(maxsize=4096)
    def neighbors(
        self, word: str, sense_id: str, relations: Optional[frozenset[Relation]] = None
    ) -> tuple[Edge, ...]:
        """Typed edges from ``word`` in sense ``sense_id``."""
        syn = self.wn.synset(sense_id)
        word_l = word.lower()
        own = next((l for l in syn.lemmas() if l.name().lower() == word_l.replace(" ", "_")), None)
        edges: list[Edge] = []

        def add(target_lemma: Any, rel: Relation) -> None:
            name = _word(target_lemma.name())
            if name.lower() == word_l:
                return
            edges.append(Edge(word_l, name, rel, sense_id, target_lemma.synset().name(),
                              float(target_lemma.count()) + 1.0))

        for l in syn.lemmas():
            add(l, Relation.SYNONYM)
        for other in syn.similar_tos() + syn.also_sees():
            for l in other.lemmas():
                add(l, Relation.SIMILAR)
        for h in syn.hypernyms():
            for l in h.lemmas():
                add(l, Relation.BROADER)
        for h in syn.hyponyms():
            for l in h.lemmas():
                add(l, Relation.NARROWER)
        if own is not None:
            for a in own.antonyms():
                add(a, Relation.ANTONYM)
            for d in own.derivationally_related_forms():
                add(d, Relation.DERIVED)

        if relations is not None:
            edges = [e for e in edges if e.relation in relations]
        return tuple(edges)

    def antonyms(self, word: str, pos: Optional[str] = None) -> set[str]:
        """Antonyms of ``word`` in ANY sense (used as a hard filter)."""
        out: set[str] = set()
        for s in self.senses(word, pos):
            for e in self.neighbors(word, s.sense_id, frozenset({Relation.ANTONYM})):
                out.add(e.target.lower())
        return out

    def relation_between(self, a: str, b: str, pos: Optional[str] = None) -> set[Relation]:
        rels: set[Relation] = set()
        for s in self.senses(a, pos):
            for e in self.neighbors(a, s.sense_id):
                if e.target.lower() == b.lower():
                    rels.add(e.relation)
        return rels

    # ── usage statistics ─────────────────────────────────────────
    @lru_cache(maxsize=8192)
    def lemma_count(self, word: str, sense_id: str) -> int:
        """How often ``word`` is used in this sense (SemCor counts)."""
        syn = self.wn.synset(sense_id)
        key = word.lower().replace(" ", "_")
        return sum(l.count() for l in syn.lemmas() if l.name().lower() == key)

    @lru_cache(maxsize=8192)
    def total_count(self, word: str, pos: Optional[str] = None) -> int:
        key = word.lower().replace(" ", "_")
        total = 0
        for syn in self.wn.synsets(key, pos=pos):
            total += sum(l.count() for l in syn.lemmas() if l.name().lower() == key)
        return total

    def dominance(self, word: str, sense_id: str, pos: Optional[str] = None) -> float:
        """Share of ``word``'s usage that is in this sense (1.0 = unambiguous).

        A low value means readers will likely understand the word in a
        *different* sense — a meaning-change risk.
        """
        total = self.total_count(word, pos)
        if total == 0:
            return 0.5 if len(self.senses(word, pos)) == 1 else 0.2
        return self.lemma_count(word, sense_id) / total

    def is_irregular_or_unknown(self, words: Iterable[str]) -> bool:  # pragma: no cover - helper
        return any(not self.senses(w) for w in words)


@lru_cache(maxsize=1)
def get_graph() -> WordGraph:
    """Shared, cached graph instance."""
    return WordGraph()
