"""Tests for the Stage 2 linguistic analyzer."""

from __future__ import annotations

import pytest

from humanizer import analyzer
from humanizer.analyzer import analyze

pytestmark = pytest.mark.skipif(
    not analyzer.is_available(),
    reason="spaCy model en_core_web_sm not installed",
)


def test_empty_text_returns_empty_analysis():
    result = analyze("   ")
    assert result.tokens == []
    assert result.sentences == []


def test_offsets_point_back_to_original_text():
    text = "Maria wrote the report.  Then she left early."
    result = analyze(text)
    for tok in result.tokens:
        assert text[tok.char_start : tok.char_end] == tok.text
    for sent in result.sentences:
        assert text[sent.char_start : sent.char_end] == sent.text


def test_sentence_boundaries():
    result = analyze("The cat sat. The dog ran. Birds sang loudly.")
    assert len(result.sentences) == 3
    assert [s.index for s in result.sentences] == [0, 1, 2]
    assert all(t.sentence_index in (0, 1, 2) for t in result.tokens)


def test_tokens_have_lemma_pos_dep_and_morph():
    result = analyze("The children were running.")
    running = next(t for t in result.tokens if t.text == "running")
    assert running.lemma == "run"
    assert running.pos == "VERB"
    children = next(t for t in result.tokens if t.text == "children")
    assert children.lemma == "child"
    assert children.morph.get("Number") == "Plur"
    assert children.dep == "nsubj"
    assert children.head_index == running.index


def test_active_clause_roles():
    result = analyze("Maria gave Tom the book.")
    clause = result.sentences[0].clauses[0]
    assert clause.subject and clause.subject.text == "Maria"
    assert clause.direct_object and clause.direct_object.text == "the book"
    assert clause.indirect_object and clause.indirect_object.text == "Tom"
    assert not clause.is_passive
    assert result.sentences[0].voice == "active"


def test_passive_clause_with_agent():
    result = analyze("The report was written by the committee.")
    clause = result.sentences[0].clauses[0]
    assert clause.is_passive
    assert clause.subject and clause.subject.text == "The report"
    assert clause.agent and clause.agent.text == "the committee"
    assert result.sentences[0].voice == "passive"


@pytest.mark.parametrize(
    "text, tense, aspect",
    [
        ("She walks to work.", "present", "simple"),
        ("She walked to work.", "past", "simple"),
        ("She will walk to work.", "future", "simple"),
        ("She is walking to work.", "present", "progressive"),
        ("She has walked to work.", "present", "perfect"),
        ("She had been walking to work.", "past", "perfect-progressive"),
    ],
)
def test_tense_and_aspect(text, tense, aspect):
    clause = analyze(text).sentences[0].clauses[0]
    assert clause.tense == tense
    assert clause.aspect == aspect


def test_modality():
    clause = analyze("Students must submit the form.").sentences[0].clauses[0]
    assert clause.modality == "necessity"
    assert clause.modal_verb == "must"


def test_question_and_imperative_mood():
    assert analyze("Did you finish it?").sentences[0].clauses[0].mood == "interrogative"
    assert analyze("Close the door.").sentences[0].clauses[0].mood == "imperative"


def test_negation_detected():
    result = analyze("The test did not pass.")
    assert any(n.cue == "not" for n in result.negations)
    assert result.sentences[0].has_negation


def test_no_false_negation():
    result = analyze("The test passed.")
    assert result.negations == []
    assert not result.sentences[0].has_negation


def test_entities_and_quantities():
    result = analyze("Apple paid $3 million to Google on March 5, 2024 for 40 kilograms of parts.")
    labels = {e.label for e in result.entities}
    assert "ORG" in labels
    kinds = {q.kind for q in result.quantities}
    assert "money" in kinds
    assert "date" in kinds
    assert any("40" in q.text for q in result.quantities)


def test_noun_phrases():
    result = analyze("The quick brown fox jumped over the lazy dog.")
    nps = [np.text for np in result.sentences[0].noun_phrases]
    assert "The quick brown fox" in nps
    assert "the lazy dog" in nps


def test_discourse_markers():
    result = analyze("It rained. However, the match continued.")
    markers = [m.text.lower() for m in result.sentences[1].discourse_markers]
    assert "however" in markers


def test_discourse_marker_not_matched_mid_clause():
    result = analyze("He finally arrived.")
    assert result.sentences[0].discourse_markers == []


def test_pronoun_antecedent_candidates():
    result = analyze("The manager met the engineer. She approved the plan.")
    she = next(p for p in result.pronouns if p.text == "She")
    assert "The manager" in she.candidates
    assert "the engineer" in she.candidates
    assert she.ambiguous


def test_repeated_words_and_openings():
    text = "The system is fast. The system is cheap. The system is reliable."
    result = analyze(text)
    assert result.repeated_words.get("system") == 3
    assert result.repeated_openings.get("the system") == 3
    assert any(f.kind == "repetition" and f.text == "system" for f in result.findings)


def test_filler_findings():
    result = analyze("We met in order to discuss the plan due to the fact that it failed.")
    fillers = {f.text.lower(): f.suggestion for f in result.findings if f.kind == "filler"}
    assert fillers["in order to"] == "to"
    assert fillers["due to the fact that"] == "because"


def test_redundant_modifier_finding():
    result = analyze("The end result was a free gift.")
    found = {f.text.lower() for f in result.findings if f.kind == "redundant_modifier"}
    assert "end result" in found
    assert "free gift" in found


def test_clean_text_has_no_redundancy_findings():
    result = analyze("The result was a gift.")
    assert not [f for f in result.findings if f.kind == "redundant_modifier"]


def test_to_dict_is_serialisable():
    import json

    json.dumps(analyze("A short sentence.").to_dict())


def test_missing_model_raises_clear_error():
    analyzer.load_nlp.cache_clear()
    with pytest.raises(analyzer.AnalyzerUnavailableError, match="spacy download"):
        analyzer.load_nlp("xx_model_that_does_not_exist")
