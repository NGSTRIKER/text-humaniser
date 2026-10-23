"""Stage 5: phrase rearrangement, combining and splitting."""

from __future__ import annotations

import pytest

from humanizer import analyzer

pytestmark = pytest.mark.skipif(not analyzer.is_available(), reason="spaCy model not installed")

from humanizer.transformations.sentence_structure import combine, split  # noqa: E402
from humanizer.transformations.word_order import back_to_front, front_to_back, move_adverb  # noqa: E402


def test_adverb_to_end():
    assert move_adverb("She carefully opened the door.") == ("She opened the door carefully.", "adverb to end")


def test_adverb_before_verb():
    assert move_adverb("She opened the door carefully.") == ("She carefully opened the door.", "adverb before verb")


@pytest.mark.parametrize("text", [
    "She only opened the door.",          # scope adverb
    "She never opened the door.",         # negation/frequency
    "She did not carefully open the door.",
    "She carefully walked.",              # no object
])
def test_adverb_not_moved(text):
    assert move_adverb(text) is None


def test_intro_phrase_to_end():
    assert front_to_back("After lunch, we met the client.") == "We met the client after lunch."


def test_conditional_clause_to_end():
    assert front_to_back("If it rains, we stay home.") == "We stay home if it rains."


def test_causal_clause_not_moved():
    assert front_to_back("Because it rained, we stayed home.") is None


def test_time_phrase_to_front():
    assert back_to_front("We met the client after lunch.") == "After lunch, we met the client."


def test_non_time_phrase_not_fronted():
    assert back_to_front("We met the client in the lobby.") is None


def test_combine_same_subject():
    assert combine("She opened the door.", "She walked in.") == "She opened the door and walked in."


def test_combine_with_unambiguous_pronoun():
    assert combine("Maria opened the door.", "She walked in.") == "Maria opened the door and walked in."


@pytest.mark.parametrize("a, b", [
    ("She opened the door.", "However, she stayed outside."),   # discourse marker
    ("She opened the door.", "He walked in."),                  # different subject
    ("Maria opened the door.", "She walked in."),               # but pronoun is ambiguous
])
def test_combine_declined(a, b):
    ambiguous = b == "She walked in."
    assert combine(a, b, ambiguous_pronoun=ambiguous) is None


def test_combine_never_adds_logical_connectors():
    out = combine("The team met.", "The team voted.")
    assert out == "The team met and voted."
    assert not any(w in out for w in ("because", "therefore", "although", "so "))


def test_split_comma_and():
    text = "The committee reviewed the proposal in great detail over three days, and the board approved the final budget on Friday."
    out, how = split(text)
    assert out == "The committee reviewed the proposal in great detail over three days. The board approved the final budget on Friday."
    assert how == "comma-and"


def test_split_semicolon():
    text = "The committee reviewed the proposal in great detail over several days; the board approved the final budget afterwards."
    assert split(text)[0].count(".") == 2


def test_short_sentence_not_split():
    assert split("We ate, and they left.") is None


def test_shared_subject_not_split():
    text = "The committee reviewed the long proposal in great detail over three days and approved the final budget on Friday."
    assert split(text) is None


def test_structural_edits_limited_per_text():
    from humanizer.config import RewriterConfig
    from humanizer.rule_engine import RuleEngine

    text = ("The report was written by the committee. After lunch, we met the client. "
            "She carefully read the report. The memo was drafted by the lawyer.")
    eng = RuleEngine(RewriterConfig())
    result = eng.run(text, intensity="moderate", seed=3)
    structural = [r for r in result.history.accepted
                  if r.transform_type in {"voice", "word_order", "sentence_structure"}]
    assert 1 <= len(structural) <= 2   # moderate: about one in three sentences
