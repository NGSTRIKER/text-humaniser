"""Stage 4 tests: word graph, morphology, sense selection, synonym rule."""

from __future__ import annotations

import pytest

from humanizer import analyzer, word_graph
from humanizer.config import RewriterConfig
from humanizer.morphology import gerund, inflect, past, plural, third_person
from humanizer.word_graph import Relation

needs_wordnet = pytest.mark.skipif(not word_graph.is_available(), reason="WordNet data not installed")
needs_all = pytest.mark.skipif(
    not (word_graph.is_available() and analyzer.is_available()), reason="WordNet or spaCy missing"
)


# ── Morphology (no external data) ───────────────────────────────


@pytest.mark.parametrize("word, expected", [
    ("cat", "cats"), ("box", "boxes"), ("city", "cities"), ("day", "days"),
    ("child", "children"), ("analysis", "analyses"),
])
def test_plural(word, expected):
    assert plural(word) == expected


@pytest.mark.parametrize("fn, word, expected", [
    (past, "walk", "walked"), (past, "decide", "decided"), (past, "try", "tried"),
    (past, "stop", "stopped"), (past, "go", "went"), (gerund, "make", "making"),
    (gerund, "run", "running"), (gerund, "lie", "lying"), (third_person, "fix", "fixes"),
    (third_person, "go", "goes"),
])
def test_verb_forms(fn, word, expected):
    assert fn(word) == expected


def test_past_participle_irregular():
    assert past("write", participle=True) == "written"


def test_inflect_by_tag():
    assert inflect("rapid", "JJ") == "rapid"
    assert inflect("finish", "VBD") == "finished"
    assert inflect("finish", "VBZ") == "finishes"
    assert inflect("set up", "VBG") == "setting up"


def test_inflect_refuses_unknown_irregular_and_comparatives():
    assert inflect("smite", "VBD", frozenset({"smite"})) is None
    assert inflect("big", "JJR") is None


# ── Word graph ───────────────────────────────────────────────────


@needs_wordnet
def test_word_has_separate_senses():
    g = word_graph.get_graph()
    senses = g.senses("bank", "n")
    assert len(senses) > 3
    assert len({s.sense_id for s in senses}) == len(senses)


@needs_wordnet
def test_relation_types_are_distinct():
    g = word_graph.get_graph()
    assert Relation.ANTONYM in g.relation_between("good", "bad", "a")
    assert Relation.SYNONYM in g.relation_between("quickly", "rapidly", "r")
    assert Relation.ANTONYM not in g.relation_between("quickly", "rapidly", "r")


@needs_wordnet
def test_antonyms_never_synonyms():
    g = word_graph.get_graph()
    assert "bad" in g.antonyms("good", "a")


@needs_wordnet
def test_dominance_flags_ambiguous_words():
    g = word_graph.get_graph()
    shares = [g.dominance("bank", s.sense_id, "n") for s in g.senses("bank", "n")]
    assert abs(sum(shares) - 1.0) < 1e-6      # shares of usage across senses
    assert min(shares) < 0.05                 # rare senses are flagged as risky
    assert max(shares) < 0.9                  # no single sense owns the word


# ── Sense selection ──────────────────────────────────────────────


@needs_wordnet
def test_lesk_picks_money_sense_of_bank():
    from humanizer.context_analyzer import disambiguate

    g = word_graph.get_graph()
    money = disambiguate(g, "bank", "n", {"money", "deposit", "loan", "account", "financial"})
    river = disambiguate(g, "bank", "n", {"river", "water", "slope", "fishing", "land"})
    assert money[0].sense.sense_id != river[0].sense.sense_id
    assert "financial" in money[0].sense.definition or "deposit" in money[0].sense.definition


def test_embedding_scorer_disables_without_vectors():
    from humanizer.context_analyzer import EmbeddingScorer

    scorer = EmbeddingScorer("xx_model_that_does_not_exist")
    assert not scorer.available
    assert scorer.similarity("big", "large") is None


# ── Synonym rule through the engine ─────────────────────────────


@pytest.fixture(scope="module")
def engine():
    from humanizer.rule_engine import RuleEngine

    return RuleEngine(RewriterConfig(deterministic=True))


@needs_all
def test_repeated_adverb_varied(engine):
    text = "The team worked quickly. Then they quickly finished."
    out = engine.run(text, seed=0).text
    assert out.startswith("The team worked quickly.")
    assert out.count("quickly") == 1
    assert "rapidly" in out or "speedily" in out


@needs_all
def test_inflection_matches_original(engine):
    text = "The results were clear. Later, the results were published."
    out = engine.run(text, seed=0).text
    second = out.split(".")[1]
    if "results" not in second:      # a swap happened → must stay plural
        assert any(w in second for w in ("effects", "outcomes"))


@needs_all
def test_no_swap_without_repetition(engine):
    text = "The team worked quickly and finished early."
    assert engine.run(text, seed=0).text == text


@needs_all
def test_light_intensity_never_swaps_words(engine):
    text = "The team worked quickly. Then they quickly finished."
    assert engine.run(text, intensity="light", seed=0).text == text


@needs_all
def test_names_and_compounds_not_swapped(engine):
    text = "Machine learning helps. Machine learning scales. Paris is big. Paris is old."
    assert engine.run(text, seed=0).text == text


@needs_all
def test_core_concept_kept(engine):
    text = "The data shows the data is good data."
    assert engine.run(text, seed=0).text == text


@needs_all
def test_connotation_blocklist(engine):
    text = "The results were important. The important thing is the results were quick."
    out = engine.run(text, seed=0).text
    assert "consequence" not in out


@needs_all
def test_ambiguous_word_rejected_and_explained(engine):
    text = "The movie was good. The music was good too."
    result = engine.run(text, seed=0)
    assert result.text == text
    assert any(r.rule_id == "synonyms.rejected" for r in result.history.records)


@needs_all
def test_synonyms_can_be_disabled():
    from humanizer.rule_engine import RuleEngine

    eng = RuleEngine(RewriterConfig(deterministic=True, synonyms_enabled=False))
    text = "The team worked quickly. Then they quickly finished."
    assert eng.run(text, seed=0).text == text


@needs_all
def test_same_seed_same_output():
    from humanizer.rule_engine import RuleEngine

    eng = RuleEngine(RewriterConfig())
    text = "The team worked quickly. Then they quickly finished and quickly left."
    assert eng.run(text, seed=5).text == eng.run(text, seed=5).text
