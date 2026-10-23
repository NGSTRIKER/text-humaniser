"""Stage 5: paragraph-level dependency analysis and opt-in reordering."""

from __future__ import annotations

import pytest

from humanizer import analyzer

pytestmark = pytest.mark.skipif(not analyzer.is_available(), reason="spaCy model not installed")

from humanizer.analyzer import analyze  # noqa: E402
from humanizer.paragraph_flow import analyze_flow, swappable_pairs  # noqa: E402


def flow(text):
    return analyze_flow(analyze(text))


def test_paragraphs_detected():
    report = flow("One sentence here. Another one.\n\nNew paragraph starts.")
    assert report.paragraphs == [[0, 1], [2]]


def test_pronoun_dependency():
    report = flow("The server crashed. The team restarted it.")
    assert report.sentences[1].depends_on_previous
    assert any("pronoun" in r for r in report.sentences[1].reasons)


def test_grammatically_complete_but_dependent():
    report = flow("The manager read the report. She approved it.")
    s = report.sentences[1]
    assert s.depends_on_previous and not s.logically_independent


def test_discourse_marker_dependency():
    report = flow("Sales fell. However, profits rose.")
    assert any("discourse marker" in r for r in report.sentences[1].reasons)


def test_chronology_detected():
    report = flow("We packed. Then we left the city.")
    assert report.has_chronology[0]


def test_demonstrative_dependency():
    report = flow("Prices doubled last year. This worried investors.")
    assert any("demonstrative" in r for r in report.sentences[1].reasons)


def test_independent_sentences():
    report = flow("Paris is in France. Water boils at 100 degrees.")
    assert not report.sentences[1].depends_on_previous


def test_first_sentence_of_paragraph_never_dependent():
    report = flow("It rained.\n\nIt was cold.")
    assert not report.sentences[1].depends_on_previous


INDEPENDENT = ("Our office opened in 2010. Paris hosts many trade fairs. Water boils at 100 degrees. "
               "Cats sleep a lot. Our office has a new website.")


def test_swappable_pairs_skip_intro_and_conclusion():
    pairs = swappable_pairs(flow(INDEPENDENT))
    assert pairs
    assert all(a != 0 and b != 4 for a, b in pairs)


def test_no_swaps_in_chronological_paragraph():
    text = "We woke up. Then we ate breakfast. Cats sleep a lot. Paris is big. Finally we left."
    assert swappable_pairs(flow(text)) == []


def test_no_swaps_in_short_paragraph():
    assert swappable_pairs(flow("Paris is big. Cats sleep. Water is wet.")) == []


def test_reordering_off_by_default():
    from humanizer.config import RewriterConfig
    from humanizer.rule_engine import RuleEngine

    eng = RuleEngine(RewriterConfig(deterministic=True))
    result = eng.run(INDEPENDENT, intensity="heavy", seed=1)
    assert not any(r.transform_type == "paragraph_flow" for r in result.history.records)


def test_reordering_when_enabled_keeps_all_sentences():
    from humanizer.config import RewriterConfig
    from humanizer.rule_engine import RuleEngine

    eng = RuleEngine(RewriterConfig(sentence_reordering=True, voice_mode="disabled", synonyms_enabled=False))
    result = eng.run(INDEPENDENT, intensity="heavy", seed=1)
    props = [r for r in result.history.records if r.transform_type == "paragraph_flow"]
    assert props
    for r in props:
        assert sorted(r.original_span.split(". ")) != [] and len(r.replacement) == len(r.original_span)
