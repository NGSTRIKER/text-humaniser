"""Writing modes, word-fit and Markdown handling."""

import pytest

from humanizer.config import RewriterConfig
from humanizer.markdown import protected_inline, split_blocks
from humanizer.styles import effective_tone, get_profile
from humanizer.word_graph import register_of

pytest.importorskip("spacy")
from humanizer.rule_engine import RuleEngine  # noqa: E402

TEXT = "The team really liked the plan. The team will present the plan next week."


@pytest.mark.parametrize("style", ["neutral", "academic", "professional", "casual"])
def test_team_never_becomes_squad_and_plan_never_program(style):
    eng = RuleEngine(RewriterConfig(style=style))
    for seed in range(8):
        for level in ("moderate", "heavy"):
            out = eng.run(TEXT, intensity=level, seed=seed).text.lower()
            assert "squad" not in out and "program" not in out


def test_registers():
    assert register_of("squad") == "informal"
    assert register_of("utilize") == "formal"
    assert register_of("demonstrate") == "elevated"
    assert register_of("team") == "neutral"


def test_academic_expands_contractions_neutral_does_not():
    t = "We don't know why it isn't working."
    acad = RuleEngine(RewriterConfig(style="academic")).run(t, intensity="light", seed=1).text
    neut = RuleEngine(RewriterConfig(style="neutral")).run(t, intensity="light", seed=1).text
    assert "do not" in acad and "is not" in acad
    assert "don't" in neut


def test_academic_instruction_forbids_inflated_words():
    tone = effective_tone("academic", None)
    assert "utilize" in tone and "own sake" in tone
    assert "Additional instruction: warm" in effective_tone("neutral", "warm")
    with pytest.raises(ValueError):
        get_profile("pirate")


def test_ornate_words_not_allowed_in_academic():
    assert get_profile("academic").allow_ornate is False


def test_markdown_blocks():
    md = "---\ntitle: x\n---\n# Head\n\nProse one.\n\n- item\n- item\n\n```\ncode\n\nmore\n```\n\n| a | b |\n\nProse two."
    blocks = split_blocks(md)
    prose = [b.text for b in blocks if b.prose]
    assert prose == ["Prose one.", "Prose two."]
    kept = [b.text for b in blocks if not b.prose]
    assert "# Head" in kept and "```\ncode\n\nmore\n```" in kept and "- item\n- item" in kept


def test_inline_protection():
    t = "See [docs](https://x.io) and `run()` or https://y.com/a."
    p = protected_inline(t)
    assert "[docs](https://x.io)" in p and "`run()`" in p
