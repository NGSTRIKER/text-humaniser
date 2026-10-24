"""Stage 3 rule tests: each rule has positive and negative cases."""

from __future__ import annotations

import pytest

from humanizer import analyzer
from humanizer.config import Intensity, RewriterConfig
from humanizer.rule_engine import RuleEngine

pytestmark = pytest.mark.skipif(not analyzer.is_available(), reason="spaCy model not installed")


@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine(RewriterConfig(deterministic=True))


def run(engine: RuleEngine, text: str, intensity: str = "moderate", **kw) -> str:
    return engine.run(text, intensity=intensity, seed=0, **kw).text


# ── Redundancy ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The end result was clear.", "The result was clear."),
        ("We will revert back to the old plan.", "We will revert to the old plan."),
        ("They completely eliminate entirely the waste.", "They eliminate the waste."),
        ("Each and every student passed.", "Every student passed."),
        ("She received a free gift.", "She received a gift."),
        ("Enter your PIN number.", "Enter your PIN."),
        ("The reason is because it rained.", "The reason is that it rained."),
    ],
)
def test_redundancy_positive(engine, text, expected):
    assert run(engine, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The result was clear.",
        "The end of the result list.",          # 'end' is not a modifier of 'result' here
        "She returned the book.",
        "The free market works.",
    ],
)
def test_redundancy_negative(engine, text):
    assert run(engine, text) == text


def test_article_fixed_after_removal(engine):
    assert run(engine, "It was an unexpected surprise.") == "It was a surprise."


# ── Wordiness ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("We left due to the fact that it rained.", "We left because it rained."),
        ("We met in order to discuss the plan.", "We met to discuss the plan."),
        ("At this point in time, we wait.", "Now, we wait."),
        ("Call me in the event that it fails.", "Call me if it fails."),
        ("It is important to note that sales rose.", "Sales rose."),
        ("The test was very slow.", "The test was slow."),
    ],
)
def test_wordiness_positive(engine, text, expected):
    assert run(engine, text) == expected


def test_in_order_to_requires_verb(engine):
    text = "Put the files in order to the left."  # not purpose 'in order to'
    assert run(engine, text) == text


def test_very_not_removed_before_noun(engine):
    text = "This is the very thing we need."
    assert run(engine, text) == text


def test_light_intensity_skips_fillers(engine):
    assert run(engine, "The test was very slow.", intensity="light") == "The test was very slow."


def test_light_still_fixes_obvious_wordiness(engine):
    assert run(engine, "We left due to the fact that it rained.", intensity="light") == "We left because it rained."


# ── Repetition ───────────────────────────────────────────────────


def test_doubled_word_removed(engine):
    assert run(engine, "We saw the the dog.") == "We saw the dog."


def test_legit_double_kept(engine):
    text = "She had had enough."
    assert run(engine, text) == text


def test_overused_word_flagged_not_changed(engine):
    text = "The data shows the data is good data."
    result = engine.run(text, seed=0)
    assert result.text == text
    flags = [r for r in result.history.records if r.rule_id == "repetition.overused_word"]
    assert flags and all(not r.accepted for r in flags)


def test_repeated_opening_flagged(engine):
    result = engine.run("The cat ate. The cat slept.", seed=0)
    assert any(r.rule_id == "repetition.sentence_opening" for r in result.history.records)


# ── Phrase transformations ──────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("We made a decision to leave.", "We decided to leave."),
        ("They make a decision every week.", "They decide every week."),
        ("She has come to a conclusion.", "She has concluded."),
        ("The police conducted an investigation into the fire.", "The police investigated the fire."),
    ],
)
def test_phrase_positive(engine, text, expected):
    assert run(engine, text) == expected


def test_phrase_off_at_light(engine):
    text = "We made a decision to leave."
    assert run(engine, text, intensity="light") == text


# ── Safeguards ───────────────────────────────────────────────────


def test_protected_term_untouched(engine):
    text = "Our product, End Result, sold well."
    assert run(engine, text, protected_terms=["End Result"]) == text


def test_negation_and_numbers_preserved(engine):
    out = run(engine, "Due to the fact that 3 people did not come, we stopped.")
    assert "not" in out and "3" in out
    assert out.startswith("Because")


# ── Modifiers ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The room was completely empty.", "The room was empty."),
        ("She received an exact replica.", "She received a replica."),
        ("They made a major breakthrough.", "They made a breakthrough."),
        ("He gave a shout loudly in the hall.", "He gave a shout in the hall."),
        ("They will merge together next month.", "They will merge next month."),
    ],
)
def test_modifiers_positive(engine, text, expected):
    assert run(engine, text) == expected


def test_clean_text_unchanged(engine):
    text = "Maria finished the report on Tuesday. Her manager approved it."
    assert run(engine, text) == text


def test_empty_text(engine):
    assert run(engine, "  ") == "  "


# ── Pipeline integration ────────────────────────────────────────


def test_pipeline_rules_only_skips_llm():
    from humanizer.rewriter import RewriterPipeline
    from humanizer.schemas import RewriteRequest

    class NoLLM:
        def generate(self, *a, **k):
            raise AssertionError("LLM must not be called in rules-only mode")

    pipe = RewriterPipeline(RewriterConfig(rules_only=True, deterministic=True), backend=NoLLM())
    result = pipe.rewrite_text(RewriteRequest(text="We left due to the fact that it rained.", seed=0))
    assert result.rewritten_text == "We left because it rained."
    assert any(e.accepted for e in result.edits)
