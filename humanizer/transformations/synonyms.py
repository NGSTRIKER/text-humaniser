"""Contextual synonym selection (Stage 4).

Only targets words that are *actually repeated* nearby; never swaps words
just for variety.  Every alternative passes through these filters:

1. Sense: Lesk-style disambiguation picks the meaning used in context;
   an unclear sense means no swap.
2. Relationship type: true synonyms (same sense) by default; near-synonyms
   add risk; broader terms only at heavy; antonyms never.
3. Part of speech + morphology: the replacement is inflected to the
   original's form and the sentence is re-parsed to confirm the POS holds.
4. Usage: the alternative must be used in this sense (SemCor count > 0)
   and not usually mean something else (dominance).
5. Register: informal/formal words are kept out of neutral text.
6. Repetition: the alternative must not already appear nearby.
7. Terminology: names, compounds ("machine learning") and the text's core
   concept noun are left alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from humanizer.analyzer import TokenInfo, load_nlp
from humanizer.candidates import Candidate
from humanizer.config import Intensity
from humanizer.context_analyzer import EmbeddingScorer, context_words, disambiguate
from humanizer.logging_config import get_logger
from humanizer.morphology import inflect
from humanizer.transformations.base import Rule, RuleContext, match_case
from humanizer.word_graph import (
    INFORMAL,
    POS_MAP,
    Relation,
    WordGraph,
    WordGraphUnavailableError,
    get_graph,
    register_of,
)

log = get_logger("synonyms")

# Very common, highly polysemous verbs: swapping them is rarely safe.
LIGHT_VERBS = frozenset(
    "be have do go get make take say give come put let see know use want need seem".split()
)


# Generic common nouns that should not be treated as protected domain terminology
GENERIC_NOUNS = frozenset({
    "thing", "way", "day", "man", "person", "people", "time", "year",
    "part", "place", "case", "point", "group", "problem", "fact",
    "feeling", "experience", "event", "situation", "issue", "individual",
    "effect", "impact", "reaction", "response", "behavior", "action",
    "thought", "emotion", "state", "aspect", "element", "area",
    "type", "kind", "form", "sort", "level", "stage", "process",
    "matter", "factor", "reason", "condition", "sense", "view",
    "life", "support", "activity", "struggle", "conflict", "pain",
})


# Curated pairs WordNet lists as synonyms but whose connotation differs
# (original lemma → alternatives that must not replace it).
NOT_INTERCHANGEABLE: dict[str, frozenset[str]] = {
    "result": frozenset({"consequence", "upshot", "issue", "event"}),
    "outcome": frozenset({"consequence", "upshot"}),
    "cheap": frozenset({"inexpensive"}),
    "inexpensive": frozenset({"cheap"}),
    "thin": frozenset({"skinny", "lean"}),
    "old": frozenset({"aged", "elderly"}),
    "smell": frozenset({"stink", "reek"}),
    "famous": frozenset({"notorious", "infamous"}),
    "confident": frozenset({"cocky", "arrogant"}),
    "child": frozenset({"kid"}),
    "house": frozenset({"family", "firm"}),
    "plan": frozenset({"program", "programme", "design"}),
    "team": frozenset({"squad", "crew", "gang"}),
    "person": frozenset({"mortal", "soul", "body", "party"}),
    "family": frozenset({"menage", "house", "syndicate", "mob", "kinfolk", "kinsfolk", "kin"}),
    "friend": frozenset({"acquaintance"}),
    "support": frozenset({"reinforcement"}),
}


@dataclass
class Option:
    word: str
    relation: Relation
    meaning_risk: float
    context_fit: float
    confidence: float
    sense_id: str


@dataclass
class OptionReport:
    """Accepted options plus why others were rejected (for explainability)."""

    sense_id: Optional[str]
    options: list[Option]
    rejected: dict[str, str]


class SynonymRule(Rule):
    rule_id = "synonyms"
    description = "Replace a repeated word with a context-appropriate synonym."
    transform_type = "synonym"
    min_intensity = Intensity.MODERATE

    def __init__(
        self,
        graph: Optional[WordGraph] = None,
        embeddings: Optional[EmbeddingScorer] = None,
        min_sense_confidence: float = 0.3,
        max_options: int = 3,
        reparse: bool = True,
        style: str = "neutral",
    ) -> None:
        from humanizer.styles import get_profile

        self.style = get_profile(style)
        self._graph = graph
        self.embeddings = embeddings
        self.min_sense_confidence = min_sense_confidence
        self.max_options = max_options
        self.reparse = reparse
        self._warned = False

    @property
    def graph(self) -> Optional[WordGraph]:
        if self._graph is None:
            try:
                self._graph = get_graph()
            except WordGraphUnavailableError as exc:
                if not self._warned:
                    log.warning("Synonym rule disabled: %s", exc)
                    self._warned = True
                return None
        return self._graph

    # ── main ─────────────────────────────────────────────────────
    def propose(self, ctx: RuleContext) -> list[Candidate]:
        if self.graph is None:
            return []
        out: list[Candidate] = []
        for tok in self.targets(ctx):
            report = self.options_for(ctx, tok)
            for opt in report.options:
                word = match_case(tok.text, opt.word)
                out.append(
                    Candidate(
                        rule_id=f"synonyms.{opt.relation.value}",
                        transform_type=self.transform_type,
                        char_start=tok.char_start, char_end=tok.char_end,
                        original=tok.text, replacement=word,
                        reason=(
                            f"Reduced repeated vocabulary: {tok.text!r} → {word!r} "
                            f"({opt.relation.value}, sense {opt.sense_id})."
                        ),
                        confidence=opt.confidence, meaning_risk=opt.meaning_risk,
                        grammar_risk=0.12, usefulness=0.55, context_fit=opt.context_fit,
                        diversity_gain=0.7,
                    )
                )
            if not report.options:
                detail = "; ".join(f"{w}: {why}" for w, why in list(report.rejected.items())[:4])
                out.append(
                    Candidate(
                        rule_id="synonyms.rejected", transform_type=self.transform_type,
                        char_start=tok.char_start, char_end=tok.char_end, original=tok.text,
                        replacement=None,
                        reason=f"Rejected synonyms for {tok.text!r} because of contextual mismatch.",
                        confidence=0.0, meaning_risk=1.0,
                        note=detail or "no alternatives in WordNet",
                    )
                )
        return out

    # ── which words to vary ──────────────────────────────────────
    def targets(self, ctx: RuleContext) -> list[TokenInfo]:
        a = ctx.analysis
        ent_spans = [(e.char_start, e.char_end) for e in a.entities]
        core = self._core_concepts(ctx)
        recent: dict[str, int] = {}   # lemma → last sentence index seen
        targets: list[TokenInfo] = []
        for tok in a.tokens:
            if tok.pos not in POS_MAP or tok.is_stop or not tok.text.isalpha():
                continue
            lemma = tok.lemma.lower()
            prev = recent.get(lemma)
            recent[lemma] = tok.sentence_index
            if prev is None:
                continue
            sent_dist = tok.sentence_index - prev
            is_repeated = lemma in a.repeated_words and a.repeated_words[lemma] >= 2
            if sent_dist > 3 and not (is_repeated and sent_dist <= 5):
                continue
            if lemma in LIGHT_VERBS or lemma in core:
                continue
            if tok.text[0].isupper() and a.sentences[tok.sentence_index].char_start != tok.char_start:
                continue  # capitalised mid-sentence → likely a name/term
            if any(s <= tok.char_start < e for s, e in ent_spans):
                continue
            if tok.dep == "compound" or any(a.tokens[c].dep == "compound" for c in tok.children):
                continue  # part of a multi-word term
            targets.append(tok)
        return targets

    @staticmethod
    def _core_concepts(ctx: RuleContext) -> set[str]:
        """Nouns used ≥3 times across ≥2 sentences are treated as terminology (unless generic)."""
        seen: dict[str, set[int]] = {}
        counts: dict[str, int] = {}
        for t in ctx.analysis.tokens:
            if t.pos == "NOUN":
                l = t.lemma.lower()
                counts[l] = counts.get(l, 0) + 1
                seen.setdefault(l, set()).add(t.sentence_index)
        return {
            l for l, c in counts.items()
            if c >= 3 and (len(seen[l]) >= 2 or c >= 3) and l not in GENERIC_NOUNS
        }

    # ── candidate generation + filtering ─────────────────────────
    def options_for(self, ctx: RuleContext, tok: TokenInfo) -> OptionReport:
        graph = self.graph
        assert graph is not None
        wn_pos = POS_MAP[tok.pos]
        lemma = tok.lemma.lower()
        cwords = context_words(ctx.analysis, tok.sentence_index)
        guesses = disambiguate(graph, lemma, wn_pos, cwords)
        if not guesses:
            return OptionReport(None, [], {lemma: "not in WordNet"})
        best = guesses[0]
        if best.confidence < self.min_sense_confidence:
            return OptionReport(
                best.sense.sense_id, [],
                {lemma: f"ambiguous sense (confidence {best.confidence:.2f})"},
            )

        allowed = {Relation.SYNONYM, Relation.SIMILAR}
        if ctx.intensity == Intensity.HEAVY:
            allowed.add(Relation.BROADER)
        antonyms = graph.antonyms(lemma, wn_pos)
        nearby = {
            t.lemma.lower() for t in ctx.analysis.tokens
            if abs(t.sentence_index - tok.sentence_index) <= 1
        }
        informal_doc = any(
            t.text.lower() in INFORMAL or t.text.lower() in {"n't", "'re", "'m", "'ll"}
            for t in ctx.analysis.tokens
        )
        irregular = frozenset(v for forms in graph.wn._exception_map["v"].values() for v in forms)

        options: list[Option] = []
        rejected: dict[str, str] = {}
        for edge in graph.neighbors(lemma, best.sense.sense_id, frozenset(allowed)):
            cand = edge.target.lower()
            why = self._reject_reason(
                ctx, graph, edge, cand, antonyms, nearby, informal_doc, wn_pos, self.style
            )
            if why:
                rejected.setdefault(cand, why)
                continue
            if tok.tag in {"NNS", "NNPS"} and cand.endswith(("ing", "ness", "ism")):
                rejected.setdefault(cand, "usually uncountable; plural would be unnatural")
                continue
            form = inflect(cand, tok.tag, irregular)
            if not form:
                rejected.setdefault(cand, f"cannot inflect to {tok.tag}")
                continue
            dominance = graph.dominance(cand, edge.target_sense_id, wn_pos)
            extra = {Relation.SYNONYM: 0.0, Relation.SIMILAR: 0.08, Relation.BROADER: 0.25}[edge.relation]
            risk = 0.06 + 0.15 * (1 - dominance) + 0.12 * (1 - best.confidence) + extra
            fit = 0.5 + 0.5 * dominance
            if register_of(cand) == "elevated":
                fit += self.style.elevated_bonus
            sim = self.embeddings.similarity(lemma, cand) if self.embeddings else None
            if sim is not None:
                fit = 0.5 * fit + 0.5 * max(0.0, sim)
            options.append(
                Option(form, edge.relation, round(min(risk, 1.0), 3), round(fit, 3),
                       round(0.5 * best.confidence + 0.5 * dominance, 3), edge.target_sense_id)
            )

        options.sort(key=lambda o: o.context_fit - o.meaning_risk, reverse=True)
        final: list[Option] = []
        for opt in options:
            if len(final) >= self.max_options:
                break
            if self.reparse and not self._pos_preserved(ctx, tok, opt.word):
                rejected.setdefault(opt.word, "changes the part of speech in context")
                continue
            final.append(opt)
        return OptionReport(best.sense.sense_id, final, rejected)

    @staticmethod
    def _reject_reason(
        ctx: RuleContext, graph: WordGraph, edge: Any, cand: str,
        antonyms: set[str], nearby: set[str], informal_doc: bool, wn_pos: str,
        style: Any = None,
    ) -> Optional[str]:
        if " " in cand or "-" in cand:
            return "multi-word alternative"
        if cand in antonyms:
            return "antonym"
        if cand in NOT_INTERCHANGEABLE.get(edge.source, frozenset()):
            return "different connotation (curated rule)"
        if cand in nearby:
            return "already used nearby (would create new repetition)"
        if graph.total_count(cand, wn_pos) > 5 and graph.lemma_count(cand, edge.target_sense_id) == 0:
            return "rarely used in this sense"
        if graph.dominance(cand, edge.target_sense_id, wn_pos) < 0.15:
            return "usually means something else"
        if edge.relation == Relation.SYNONYM:
            # Same meaning but a much rarer word for it ("team" 43 vs "squad" 4
            # in WordNet's counts) usually sounds off: skip it.
            src = graph.lemma_count(edge.source, edge.target_sense_id)
            if src > 10 and graph.total_count(cand, wn_pos) > 5 and graph.lemma_count(cand, edge.target_sense_id) < 0.25 * src:
                return "much less common word for this meaning"
        reg = register_of(cand)
        casual = bool(style and style.allow_informal)
        formal_style = bool(style and style.name in ("professional", "academic"))
        if reg == "informal" and (formal_style or not (informal_doc or casual)):
            return "too informal for this text"
        if reg == "formal" and (formal_style or casual or ctx.intensity != Intensity.HEAVY):
            return "overly ornate word"
        if reg == "elevated" and casual:
            return "too formal for casual style"
        return None

    @staticmethod
    def _pos_preserved(ctx: RuleContext, tok: TokenInfo, word: str) -> bool:
        """Re-parse the sentence with the replacement and compare POS."""
        sent = ctx.analysis.sentences[tok.sentence_index]
        rel = tok.char_start - sent.char_start
        new_sent = sent.text[:rel] + word + sent.text[rel + len(tok.text):]
        try:
            doc = load_nlp()(new_sent)
        except Exception:  # pragma: no cover - analyzer already ran, so unlikely
            return False
        new_tok = next((t for t in doc if t.idx == rel), None)
        return new_tok is not None and new_tok.pos_ == tok.pos
