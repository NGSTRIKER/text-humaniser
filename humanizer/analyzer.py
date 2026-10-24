"""Stage 2 — Linguistic analyzer built on spaCy.

Turns raw text into typed, offset-preserving analysis objects that later
stages (rule engine, voice transforms, validator) consume.

Design notes:
    * Every span keeps ``char_start``/``char_end`` into the ORIGINAL text so
      edits can always be traced back to their source.
    * Parser output is *evidence*, not truth.  Anything heuristic carries a
      ``confidence`` so downstream code can skip risky transformations.
    * The spaCy model is loaded lazily and cached, so importing this module
      is cheap and the rest of the program works even if spaCy is missing.

Usage:
    >>> from humanizer.analyzer import analyze
    >>> doc = analyze("The report was written by Maria in 2024.")
    >>> doc.sentences[0].voice
    'passive'

Debug from the command line:
    python -m humanizer.analyzer "Some text to inspect."
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "en_core_web_sm"


class AnalyzerUnavailableError(RuntimeError):
    """Raised when spaCy or its English model is not installed."""


# ── Word lists (small, conservative, easy to extend) ─────────────

NEGATION_WORDS = frozenset(
    {"not", "n't", "no", "never", "none", "nobody", "nothing", "neither",
     "nor", "nowhere", "without", "cannot", "hardly", "barely", "scarcely"}
)

MODAL_MEANINGS: dict[str, str] = {
    "can": "ability", "could": "possibility", "may": "possibility",
    "might": "possibility", "must": "necessity", "shall": "obligation",
    "should": "obligation", "will": "prediction", "would": "conditional",
    "ought": "obligation",
}

DISCOURSE_MARKERS = (
    "however", "therefore", "moreover", "furthermore", "meanwhile",
    "consequently", "nevertheless", "nonetheless", "additionally", "thus",
    "hence", "instead", "otherwise", "similarly", "likewise", "finally",
    "first", "second", "third", "also", "besides", "indeed", "still",
    "in addition", "for example", "for instance", "on the other hand",
    "as a result", "in contrast", "in fact", "in other words", "that said",
    "after all", "of course", "in conclusion", "to summarize",
)

FILLER_PHRASES: dict[str, str] = {
    "in order to": "to",
    "due to the fact that": "because",
    "owing to the fact that": "because",
    "at this point in time": "now",
    "at the present time": "now",
    "in the event that": "if",
    "for the purpose of": "for",
    "with regard to": "about",
    "with respect to": "about",
    "in spite of the fact that": "although",
    "it is important to note that": "",
    "it should be noted that": "",
    "the fact that": "that",
    "a large number of": "many",
    "a majority of": "most",
    "has the ability to": "can",
    "is able to": "can",
    "in the process of": "",
    "basically": "",
    "actually": "",
    "really": "",
    "very": "",
    "quite": "",
    "just": "",
}

# adjective/adverb + head where the modifier adds nothing ("end result")
REDUNDANT_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {("end", "result"), ("past", "history"), ("future", "plan"),
     ("free", "gift"), ("basic", "fundamental"), ("final", "outcome"),
     ("advance", "planning"), ("close", "proximity"), ("completely", "finish"),
     ("absolutely", "essential"), ("unexpected", "surprise"),
     ("true", "fact"), ("each", "individual"), ("new", "innovation"),
     ("added", "bonus"), ("joint", "collaboration"), ("brief", "summary"),
     ("repeat", "again"), ("revert", "back"), ("return", "back")}
)

STOP_LEMMAS = frozenset({"be", "have", "do", "the", "a", "an", "and", "of", "to", "in", "-PRON-"})


# ── Typed result structures ──────────────────────────────────────


@dataclass(frozen=True)
class Span:
    """A piece of the original text with its exact character offsets."""

    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class TokenInfo:
    """One token with everything spaCy knows about it."""

    index: int               # token index in the whole document
    text: str
    lemma: str
    pos: str                 # coarse universal POS (NOUN, VERB…)
    tag: str                 # fine-grained tag (VBD, NNS…)
    dep: str                 # dependency label
    head_index: int
    children: tuple[int, ...]
    morph: dict[str, str]
    char_start: int
    char_end: int
    sentence_index: int
    is_stop: bool
    is_punct: bool
    like_num: bool
    whitespace: str


@dataclass(frozen=True)
class EntityInfo:
    """A named entity (PERSON, ORG, DATE, MONEY …)."""

    text: str
    label: str
    char_start: int
    char_end: int
    sentence_index: int


@dataclass(frozen=True)
class QuantityInfo:
    """A number, date, quantity or measurement that must be preserved."""

    text: str
    kind: str                # number | date | time | money | percent | quantity | ordinal
    char_start: int
    char_end: int
    unit: Optional[str] = None


@dataclass(frozen=True)
class NegationInfo:
    """A negation cue and the word it negates."""

    cue: str
    negated: str
    char_start: int
    char_end: int
    sentence_index: int


@dataclass(frozen=True)
class ClauseInfo:
    """A clause headed by a verb, with its grammatical roles."""

    verb: str
    verb_lemma: str
    verb_index: int
    clause_type: str         # main | relative | adverbial | complement | coordinate | other
    subject: Optional[Span] = None
    direct_object: Optional[Span] = None
    indirect_object: Optional[Span] = None
    agent: Optional[Span] = None       # "by X" in passive clauses
    is_passive: bool = False
    negated: bool = False
    tense: Optional[str] = None        # past | present | future
    aspect: Optional[str] = None       # simple | progressive | perfect | perfect-progressive
    mood: Optional[str] = None         # indicative | imperative | interrogative
    modality: Optional[str] = None     # ability | possibility | necessity …
    modal_verb: Optional[str] = None
    confidence: float = 1.0


@dataclass(frozen=True)
class PronounInfo:
    """A pronoun and the noun phrases that might be its antecedent."""

    text: str
    char_start: int
    char_end: int
    sentence_index: int
    candidates: tuple[str, ...]
    ambiguous: bool


@dataclass(frozen=True)
class Finding:
    """A heuristic flag (filler phrase, redundant modifier, repetition)."""

    kind: str
    text: str
    char_start: int
    char_end: int
    suggestion: Optional[str] = None
    confidence: float = 0.5


@dataclass
class SentenceAnalysis:
    """Everything known about one sentence."""

    index: int
    text: str
    char_start: int
    char_end: int
    token_indices: tuple[int, ...]
    root_lemma: str
    opening: str                       # first 1–2 words, for variety checks
    opening_pattern: str               # POS of first two tokens, e.g. "DET NOUN"
    noun_phrases: list[Span] = field(default_factory=list)
    verb_phrases: list[Span] = field(default_factory=list)
    clauses: list[ClauseInfo] = field(default_factory=list)
    discourse_markers: list[Span] = field(default_factory=list)
    conjunctions: list[Span] = field(default_factory=list)
    is_question: bool = False
    voice: str = "active"              # active | passive | mixed | none

    @property
    def has_negation(self) -> bool:
        return any(c.negated for c in self.clauses)


@dataclass
class DocumentAnalysis:
    """Top-level analysis result for a text."""

    text: str
    model: str
    tokens: list[TokenInfo] = field(default_factory=list)
    sentences: list[SentenceAnalysis] = field(default_factory=list)
    entities: list[EntityInfo] = field(default_factory=list)
    quantities: list[QuantityInfo] = field(default_factory=list)
    negations: list[NegationInfo] = field(default_factory=list)
    pronouns: list[PronounInfo] = field(default_factory=list)
    repeated_words: dict[str, int] = field(default_factory=dict)
    repeated_phrases: dict[str, int] = field(default_factory=dict)
    repeated_openings: dict[str, int] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (JSON-serialisable) for logging and debugging."""
        return asdict(self)

    def span_text(self, start: int, end: int) -> str:
        return self.text[start:end]


# ── spaCy loading ────────────────────────────────────────────────


@lru_cache(maxsize=4)
def _load_nlp_cached(model: str) -> Any:
    """Internal cached loader for a spaCy pipeline."""
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - depends on env
        raise AnalyzerUnavailableError(
            "spaCy is not installed. Run: uv add spacy"
        ) from exc
    try:
        nlp = spacy.load(model)
    except OSError as exc:
        if model in {"en_core_web_sm", "en_core_web_md", "en_core_web_lg"}:
            try:
                import spacy.cli

                logger.info("spaCy model '%s' not found; attempting automatic download...", model)
                spacy.cli.download(model)
                nlp = spacy.load(model)
                logger.info("Loaded spaCy model %s", model)
                return nlp
            except (SystemExit, Exception):
                pass
        raise AnalyzerUnavailableError(
            f"spaCy model '{model}' is not installed. "
            f"Run: uv run python -m spacy download {model}"
        ) from exc
    logger.info("Loaded spaCy model %s", model)
    return nlp


def load_nlp(model: Optional[str] = None) -> Any:
    """Load (and cache) a spaCy pipeline.

    Normalizes default model parameter so the model is never loaded twice.

    Raises:
        AnalyzerUnavailableError: with install instructions if missing.
    """
    target = model or DEFAULT_MODEL
    return _load_nlp_cached(target)


load_nlp.cache_clear = _load_nlp_cached.cache_clear  # type: ignore[attr-defined]
load_nlp.cache_info = _load_nlp_cached.cache_info  # type: ignore[attr-defined]


def is_available(model: Optional[str] = None) -> bool:
    """True if spaCy and the model can be loaded."""
    try:
        load_nlp(model)
        return True
    except AnalyzerUnavailableError:
        return False


# ── Helpers ──────────────────────────────────────────────────────


def _span(tokens: Any) -> Span:
    """Span covering a spaCy Span or list of tokens (contiguous range)."""
    toks = list(tokens)
    start, end = toks[0].idx, toks[-1].idx + len(toks[-1].text)
    return Span(toks[0].doc.text[start:end], start, end)


def _subtree_span(tok: Any) -> Span:
    return _span(tok.doc[tok.left_edge.i : tok.right_edge.i + 1])


def _clause_type(verb: Any) -> str:
    dep = verb.dep_
    if dep == "ROOT":
        return "main"
    if dep == "relcl":
        return "relative"
    if dep == "advcl":
        return "adverbial"
    if dep in {"ccomp", "xcomp", "csubj"}:
        return "complement"
    if dep == "conj":
        return "coordinate"
    return "other"


def _verb_group(verb: Any) -> list[Any]:
    """Auxiliaries + negation + main verb, e.g. 'has not been written'."""
    parts = [c for c in verb.children if c.dep_ in {"aux", "auxpass", "neg"}]
    return sorted(parts + [verb], key=lambda t: t.i)


def _tense_aspect(verb: Any) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Infer tense, aspect, modality and modal verb from the verb group."""
    group = _verb_group(verb)
    auxes = [t for t in group if t.dep_ in {"aux", "auxpass"}]
    lemmas = [t.lemma_.lower() for t in auxes]
    modal = next((t for t in auxes if t.tag_ == "MD"), None)
    modal_word = modal.text.lower() if modal else None
    modality = MODAL_MEANINGS.get(modal.lemma_.lower(), "other") if modal else None

    perfect = "have" in lemmas and any(t.tag_ == "VBN" for t in group)
    progressive = any(t.tag_ == "VBG" for t in group) and "be" in lemmas
    aspect = (
        "perfect-progressive" if perfect and progressive
        else "perfect" if perfect
        else "progressive" if progressive
        else "simple"
    )

    if modal_word in {"will", "shall"} or (
        len(group) > 1 and group[0].lower_ == "going" and "to" in [t.lower_ for t in group]
    ):
        tense = "future"
    else:
        finite = next((t for t in group if t.tag_ in {"VBD", "VBZ", "VBP", "MD"}), None)
        if finite is None:
            tense = None
        elif finite.tag_ == "VBD" or finite.lower_ in {"could", "would", "might", "should"}:
            tense = "past" if finite.tag_ == "VBD" else "present"
        else:
            tense = "present"
    return tense, aspect, modality, modal_word


def _quantity_kind(label: str) -> Optional[str]:
    return {
        "DATE": "date", "TIME": "time", "MONEY": "money", "PERCENT": "percent",
        "QUANTITY": "quantity", "CARDINAL": "number", "ORDINAL": "ordinal",
    }.get(label)


@lru_cache(maxsize=256)
def _phrase_regex(phrase: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.IGNORECASE)


def _find_phrase(text: str, phrase: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _phrase_regex(phrase).finditer(text)]


def _model_name(nlp: Any, fallback: str) -> str:
    meta = getattr(nlp, "meta", {}) or {}
    return f"{meta['lang']}_{meta['name']}" if "lang" in meta and "name" in meta else fallback


# ── Main entry point ─────────────────────────────────────────────


def analyze(text: str, model: str = DEFAULT_MODEL, nlp: Any = None) -> DocumentAnalysis:
    """Run full linguistic analysis on ``text``.

    Args:
        text: Input text (any length; spaCy handles paragraphs).
        model: spaCy model name, used if ``nlp`` is not supplied.
        nlp: An already-loaded spaCy pipeline (handy for tests).

    Returns:
        A :class:`DocumentAnalysis` with offsets into ``text``.
    """
    nlp = nlp or load_nlp(model)
    result = DocumentAnalysis(text=text, model=_model_name(nlp, model))
    if not text.strip():
        return result

    doc = nlp(text)
    sent_of: dict[int, int] = {}
    for si, sent in enumerate(doc.sents):
        for tok in sent:
            sent_of[tok.i] = si

    # tokens
    for tok in doc:
        result.tokens.append(
            TokenInfo(
                index=tok.i, text=tok.text, lemma=tok.lemma_, pos=tok.pos_, tag=tok.tag_,
                dep=tok.dep_, head_index=tok.head.i,
                children=tuple(c.i for c in tok.children),
                morph=tok.morph.to_dict(), char_start=tok.idx,
                char_end=tok.idx + len(tok.text), sentence_index=sent_of[tok.i],
                is_stop=tok.is_stop, is_punct=tok.is_punct, like_num=tok.like_num,
                whitespace=tok.whitespace_,
            )
        )

    # entities and quantities
    for ent in doc.ents:
        result.entities.append(
            EntityInfo(ent.text, ent.label_, ent.start_char, ent.end_char, sent_of[ent.start])
        )
        kind = _quantity_kind(ent.label_)
        if kind:
            unit = None
            if kind == "quantity":
                nouns = [t.text for t in ent if t.pos_ in {"NOUN", "PROPN"}]
                unit = nouns[-1] if nouns else None
            result.quantities.append(
                QuantityInfo(ent.text, kind, ent.start_char, ent.end_char, unit)
            )
    covered = {i for q in result.quantities for i in range(q.char_start, q.char_end)}
    for tok in doc:
        if tok.like_num and tok.idx not in covered:
            unit = tok.head.text if tok.dep_ == "nummod" and tok.head.pos_ == "NOUN" else None
            result.quantities.append(
                QuantityInfo(tok.text, "number", tok.idx, tok.idx + len(tok.text), unit)
            )

    # negation
    for tok in doc:
        if tok.dep_ == "neg" or tok.lower_ in NEGATION_WORDS:
            result.negations.append(
                NegationInfo(tok.text, tok.head.text, tok.idx, tok.idx + len(tok.text), sent_of[tok.i])
            )

    # sentences
    lower_text = text.lower()
    for si, sent in enumerate(doc.sents):
        result.sentences.append(_analyze_sentence(si, sent, lower_text))

    # pronouns and possible antecedents (previous + current sentence)
    for tok in doc:
        if tok.pos_ != "PRON" or tok.lower_ in {"i", "me", "you", "we", "us", "my", "your", "our"}:
            continue
        si = sent_of[tok.i]
        plural = tok.morph.get("Number") == ["Plur"]
        cands: list[str] = []
        for s in result.sentences[max(0, si - 1) : si + 1]:
            for np_ in s.noun_phrases:
                if np_.char_start >= tok.idx:
                    continue
                head = doc.char_span(np_.char_start, np_.char_end)
                if head is None or head.root.pos_ == "PRON":
                    continue
                np_plural = head.root.tag_ in {"NNS", "NNPS"}
                if np_plural == plural:
                    cands.append(np_.text)
        result.pronouns.append(
            PronounInfo(tok.text, tok.idx, tok.idx + len(tok.text), si,
                        tuple(dict.fromkeys(cands)), len(set(cands)) > 1)
        )

    # repetition
    lemmas = Counter(
        t.lemma_.lower() for t in doc
        if t.is_alpha and not t.is_stop and t.lemma_.lower() not in STOP_LEMMAS
    )
    result.repeated_words = {w: c for w, c in lemmas.most_common() if c > 1}
    phrases = Counter(
        " ".join(t.lower_ for t in doc[i : i + n])
        for n in (2, 3) for i in range(len(doc) - n + 1)
        if all(t.is_alpha for t in doc[i : i + n]) and not all(t.is_stop for t in doc[i : i + n])
    )
    result.repeated_phrases = {p: c for p, c in phrases.most_common() if c > 1}
    openings = Counter(s.opening.lower() for s in result.sentences if s.opening)
    result.repeated_openings = {o: c for o, c in openings.items() if c > 1}

    # heuristic findings
    for phrase, suggestion in FILLER_PHRASES.items():
        for start, end in _find_phrase(text, phrase):
            result.findings.append(
                Finding("filler", text[start:end], start, end, suggestion,
                        0.8 if len(phrase.split()) > 1 else 0.4)
            )
    for tok in doc:
        pair = (tok.lemma_.lower(), tok.head.lemma_.lower())
        if tok.dep_ in {"amod", "advmod", "compound", "prt"} and (pair in REDUNDANT_PAIRS or pair[::-1] in REDUNDANT_PAIRS):
            a, b = sorted([tok, tok.head], key=lambda t: t.i)
            start, end = a.idx, b.idx + len(b.text)
            result.findings.append(
                Finding("redundant_modifier", text[start:end], start, end, tok.head.text, 0.7)
            )
    for word, count in result.repeated_words.items():
        if count >= 3:
            first = next(t for t in doc if t.lemma_.lower() == word)
            result.findings.append(
                Finding("repetition", word, first.idx, first.idx + len(first.text), None,
                        min(0.9, 0.3 + 0.15 * count))
            )
    result.findings.sort(key=lambda f: f.char_start)
    return result


def _analyze_sentence(si: int, sent: Any, lower_text: str) -> SentenceAnalysis:
    words = [t for t in sent if not t.is_punct and not t.is_space]
    opening = " ".join(t.text for t in words[:2]) if words else ""
    analysis = SentenceAnalysis(
        index=si, text=sent.text, char_start=sent.start_char, char_end=sent.end_char,
        token_indices=tuple(t.i for t in sent), root_lemma=sent.root.lemma_,
        opening=opening, opening_pattern=" ".join(t.pos_ for t in words[:2]),
        is_question=sent.text.rstrip().endswith("?"),
    )

    analysis.noun_phrases = [_span(nc) for nc in sent.noun_chunks]
    analysis.conjunctions = [
        Span(t.text, t.idx, t.idx + len(t.text)) for t in sent if t.pos_ in {"CCONJ", "SCONJ"}
    ]

    sent_lower = lower_text[sent.start_char : sent.end_char]
    for marker in DISCOURSE_MARKERS:
        for start, end in _find_phrase(sent_lower, marker):
            # only count as discourse marker at clause start or after comma/semicolon
            before = sent_lower[:start].rstrip()
            if before == "" or before.endswith((",", ";", ":")):
                abs_s = sent.start_char + start
                analysis.discourse_markers.append(
                    Span(sent.doc.text[abs_s : abs_s + end - start], abs_s, abs_s + end - start)
                )

    verbs = [t for t in sent if t.pos_ in {"VERB", "AUX"} and t.dep_ not in {"aux", "auxpass"}]
    for verb in verbs:
        group = _verb_group(verb)
        analysis.verb_phrases.append(_span(group) if _contiguous(group) else Span(verb.text, verb.idx, verb.idx + len(verb.text)))
        analysis.clauses.append(_clause(verb, analysis.is_question))

    kinds = {c.is_passive for c in analysis.clauses}
    analysis.voice = (
        "none" if not kinds else "mixed" if len(kinds) == 2
        else "passive" if True in kinds else "active"
    )
    return analysis


def _contiguous(tokens: list[Any]) -> bool:
    return all(b.i - a.i == 1 for a, b in zip(tokens, tokens[1:]))


def _clause(verb: Any, is_question: bool) -> ClauseInfo:
    subj = obj = iobj = agent = None
    passive = False
    confidence = 1.0
    for c in verb.children:
        if c.dep_ in {"nsubj", "csubj", "expl"}:
            subj = _subtree_span(c)
        elif c.dep_ in {"nsubjpass", "csubjpass"}:
            subj, passive = _subtree_span(c), True
        elif c.dep_ in {"dobj", "obj"}:
            obj = _subtree_span(c)
        elif c.dep_ == "dative":
            iobj = _subtree_span(c)
        elif c.dep_ == "auxpass":
            passive = True
        elif c.dep_ == "agent":
            pobj = [g for g in c.children if g.dep_ == "pobj"]
            if pobj:
                agent = _subtree_span(pobj[0])
    if subj is None and verb.dep_ == "conj":
        # shared subject: "She wrote and published the book."
        for c in verb.head.children:
            if c.dep_ in {"nsubj", "nsubjpass"}:
                subj = _subtree_span(c)
                confidence = 0.7
    if subj is None and verb.dep_ != "ROOT":
        confidence = min(confidence, 0.6)

    tense, aspect, modality, modal = _tense_aspect(verb)
    if is_question:
        mood = "interrogative"
    elif verb.dep_ == "ROOT" and subj is None and verb.tag_ == "VB":
        mood = "imperative"
    else:
        mood = "indicative"
    negated = any(c.dep_ == "neg" for c in verb.children) or any(
        c.lower_ in NEGATION_WORDS for c in verb.children if c.dep_ in {"advmod", "det"}
    )
    return ClauseInfo(
        verb=verb.text, verb_lemma=verb.lemma_, verb_index=verb.i,
        clause_type=_clause_type(verb), subject=subj, direct_object=obj,
        indirect_object=iobj, agent=agent, is_passive=passive, negated=negated,
        tense=tense, aspect=aspect, mood=mood, modality=modality, modal_verb=modal,
        confidence=confidence,
    )


# ── CLI for debugging ────────────────────────────────────────────


def _main(argv: list[str]) -> int:
    text = " ".join(argv) if argv else sys.stdin.read()
    try:
        result = analyze(text)
    except AnalyzerUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    data = result.to_dict()
    data.pop("tokens")  # very long; omit from the summary view
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
