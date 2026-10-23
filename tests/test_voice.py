"""Stage 5 voice conversion tests (replaces the Stage 1 stub spec)."""

from __future__ import annotations

import pytest

from humanizer import analyzer

pytestmark = pytest.mark.skipif(not analyzer.is_available(), reason="spaCy model not installed")

from humanizer.transformations.voice import should_convert, to_active, to_passive  # noqa: E402

ACTIVE_TO_PASSIVE = [
    ("simple_past", "John threw rocks into the lake.", "Rocks were thrown into the lake by John."),
    ("simple_present", "The cat chases the mouse.", "The mouse is chased by the cat."),
    ("present_perfect", "John has thrown the ball.", "The ball has been thrown by John."),
    ("modal_verb", "John might throw the ball.", "The ball might be thrown by John."),
    ("negation", "John did not throw the ball.", "The ball was not thrown by John."),
    ("future_tense", "The team will complete the project.", "The project will be completed by the team."),
    ("contracted_negation", "They didn't sign the contract.", "The contract wasn't signed by them."),
    ("progressive", "The kids are painting the fence.", "The fence is being painted by the kids."),
    ("plural_object_present", "The teacher grades the essays.", "The essays are graded by the teacher."),
    ("adverb_kept", "She carefully opened the door.", "The door was carefully opened by her."),
    ("object_pronoun", "The manager praised him.", "He was praised by the manager."),
]

PASSIVE_TO_ACTIVE = [
    ("simple_passive", "The experiment was conducted by the research team.", "The research team conducted the experiment."),
    ("past_plural", "Rocks were thrown into the lake by John.", "John threw rocks into the lake."),
    ("perfect", "The ball has been thrown by John.", "John has thrown the ball."),
    ("modal", "The ball might be thrown by John.", "John might throw the ball."),
    ("negation", "The ball was not thrown by John.", "John did not throw the ball."),
    ("present", "The mouse is chased by the cat.", "The cat chases the mouse."),
    ("progressive", "The fence is being painted by the kids.", "The kids are painting the fence."),
    ("pronoun_case", "The letters were written by her.", "She wrote the letters."),
    ("present_negative", "The reports are not checked by the manager.", "The manager does not check the reports."),
]


@pytest.mark.parametrize("cid, active, passive", ACTIVE_TO_PASSIVE, ids=[c[0] for c in ACTIVE_TO_PASSIVE])
def test_active_to_passive(cid, active, passive):
    assert to_passive(active).text == passive


@pytest.mark.parametrize("cid, active, passive", ACTIVE_TO_PASSIVE[:6], ids=[c[0] for c in ACTIVE_TO_PASSIVE[:6]])
def test_roundtrip_restores_original(cid, active, passive):
    assert to_active(to_passive(active).text).text == active


@pytest.mark.parametrize("cid, passive, active", PASSIVE_TO_ACTIVE, ids=[c[0] for c in PASSIVE_TO_ACTIVE])
def test_passive_to_active(cid, passive, active):
    assert to_active(passive).text == active


@pytest.mark.parametrize("text", ["The results were analyzed.", "Mistakes were made."])
def test_agentless_passive_declined(text):
    res = to_active(text)
    assert res.text is None
    assert "agent" in res.note


@pytest.mark.parametrize("text", [
    "I went to the store and bought some groceries.",   # conversational, multi-clause
    "Mistakes were made.",                              # intentionally agentless
    "We love this song.",                               # first person
    "She has a car.",                                   # stative verb
    "Did John throw the ball?",                         # question
    "John gave Mary the book.",                         # ditransitive
])
def test_should_not_convert(text):
    assert should_convert(text) is False


def test_tense_preserved():
    out = to_passive("John throws the ball.").text
    assert out == "The ball is thrown by John."
    assert "was" not in out


def test_negation_preserved():
    assert " not " in to_passive("John did not throw the ball.").text


def test_modality_preserved():
    assert "might" in to_passive("John might throw the ball.").text
    assert "must" in to_passive("The staff must clean the kitchen.").text


def test_aspect_preserved():
    assert "has been" in to_passive("John has thrown the ball.").text
    assert "had been" in to_passive("The board had approved the plan.").text


def test_agreement_with_new_subject():
    assert to_passive("The chef cooks the meals.").text.startswith("The meals are")
    assert to_passive("The chefs cook the meal.").text.startswith("The meal is")


def test_participants_preserved():
    out = to_passive("Maria sent the invoice to the client.").text
    for part in ("Maria", "the invoice", "to the client"):
        assert part.lower() in out.lower()


def test_rule_records_voice_change_in_history():
    from humanizer.config import RewriterConfig
    from humanizer.rule_engine import RuleEngine

    eng = RuleEngine(RewriterConfig(deterministic=True))
    result = eng.run("The experiment was conducted by the research team.", seed=0)
    assert result.text == "The research team conducted the experiment."
    rec = next(r for r in result.history.accepted if r.transform_type == "voice")
    assert "passive voice to active" in rec.reason


def test_voice_disabled():
    from humanizer.config import RewriterConfig
    from humanizer.rule_engine import RuleEngine

    eng = RuleEngine(RewriterConfig(deterministic=True, voice_mode="disabled"))
    text = "The experiment was conducted by the research team."
    assert eng.run(text, seed=0).text == text


def test_light_never_converts_voice():
    from humanizer.config import RewriterConfig
    from humanizer.rule_engine import RuleEngine

    eng = RuleEngine(RewriterConfig(deterministic=True))
    text = "The experiment was conducted by the research team."
    assert eng.run(text, intensity="light", seed=0).text == text
