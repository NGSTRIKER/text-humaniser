"""Stage 6: probabilistic selection, intensity profiles, seeds, conflicts."""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from humanizer.candidates import Candidate
from humanizer.config import Intensity, RewriterConfig
from humanizer.conflicts import conflict
from humanizer.probability import derive_seed, selection_probabilities
from humanizer.profiles import resolve_profile
from humanizer.weighting import gate, score, select


def cand(start, end, rule_id="r", ttype="wordiness", original="x", replacement="y", **kw):
    return Candidate(rule_id, ttype, start, end, original, replacement, "test", **kw)


# ── probabilities ────────────────────────────────────────────────
@pytest.mark.parametrize("method", ["softmax", "linear", "uniform"])
def test_probabilities_sum_to_one(method):
    p = selection_probabilities([0.2, 0.9, 0.5], method, 0.5)
    assert sum(p) == pytest.approx(1.0)


def test_softmax_prefers_higher_scores_and_temperature_flattens():
    cold = selection_probabilities([0.2, 1.0], "softmax", 0.1)
    hot = selection_probabilities([0.2, 1.0], "softmax", 5.0)
    assert cold[1] > 0.99
    assert hot[1] < 0.6


def test_bad_method_or_temperature_rejected():
    with pytest.raises(ValueError):
        selection_probabilities([1.0], "softmax", 0)
    with pytest.raises(ValueError):
        selection_probabilities([1.0], "bogus")  # type: ignore[arg-type]


# ── seeds ────────────────────────────────────────────────────────
def test_derived_seed_is_stable_and_key_dependent():
    assert derive_seed(42, 0) == derive_seed(42, 0)
    assert derive_seed(42, 0) != derive_seed(42, 1)
    assert derive_seed(None, 3) is None


def test_selection_frequencies_follow_scores():
    cfg = RewriterConfig(selection_temperature=0.3)
    counts = Counter()
    rng = random.Random(1)
    for _ in range(400):
        lo, hi = cand(0, 3, "lo", usefulness=0.3), cand(1, 4, "hi", usefulness=0.9)
        for c in (lo, hi):
            score(c, cfg)
        counts[select([lo, hi], cfg, 1, rng)[0][0].rule_id] += 1
    assert counts["hi"] > counts["lo"] > 0  # varied, but biased to the better edit


def test_selected_edit_records_probability():
    cfg = RewriterConfig()
    cs = [cand(i * 10, i * 10 + 3, str(i)) for i in range(3)]
    for c in cs:
        score(c, cfg)
    selected, _ = select(cs, cfg, 1, random.Random(0))
    assert 0 < selected[0].selection_probability <= 1


# ── gates beat randomness ────────────────────────────────────────
def test_high_risk_never_selected_regardless_of_seed():
    cfg = RewriterConfig(randomness=1.0)
    for seed in range(20):
        risky = cand(0, 3, "risky", meaning_risk=0.9, usefulness=1.0)
        assert gate(risky, cfg, Intensity.HEAVY, []) is not None


def test_light_profile_disables_structural_edits():
    cfg = RewriterConfig()
    reason = gate(cand(0, 5, ttype="voice"), cfg, Intensity.LIGHT, [])
    assert reason and "disabled" in reason
    assert gate(cand(0, 5, ttype="voice"), cfg, Intensity.MODERATE, []) is None


def test_domain_risk_lowers_score():
    cfg = RewriterConfig()
    assert score(cand(0, 3, domain_risk=0.5), cfg) < score(cand(0, 3), cfg)


# ── profiles and user controls ───────────────────────────────────
def test_profiles_scale_with_intensity():
    cfg = RewriterConfig()
    light, heavy = resolve_profile(Intensity.LIGHT, cfg), resolve_profile(Intensity.HEAVY, cfg)
    assert light.edit_cap(200) < heavy.edit_cap(200)
    assert light.max_meaning_risk < heavy.max_meaning_risk
    assert light.temperature < heavy.temperature
    assert light.structural_cap(10) == 0 < heavy.structural_cap(10)


def test_user_overrides_apply():
    cfg = RewriterConfig(max_edits=2, randomness=0.0, type_weights={"synonym": 0.0}, restructuring="low")
    p = resolve_profile(Intensity.HEAVY, cfg)
    assert p.edit_cap(1000) == 2
    assert p.temperature == pytest.approx(0.05)
    assert p.multiplier("synonym") == 0.0
    assert p.multiplier("voice") < 1.0


def test_max_edit_ratio_caps_rate():
    p = resolve_profile(Intensity.HEAVY, RewriterConfig(max_edit_ratio=0.05))
    assert p.edit_cap(100) == 5


def test_negative_type_weight_rejected():
    with pytest.raises(ValueError):
        RewriterConfig(type_weights={"voice": -1})


def test_weights_file(tmp_path):
    f = tmp_path / "w.json"
    f.write_text(json.dumps({"weight_meaning_risk": 2.5, "type_weights": {"voice": 0.5}}))
    cfg = RewriterConfig.from_weights_file(f, seed=3)
    assert cfg.weight_meaning_risk == 2.5 and cfg.type_weights == {"voice": 0.5} and cfg.seed == 3
    f.write_text(json.dumps({"weight_typo": 1}))
    with pytest.raises(ValueError, match="unknown"):
        RewriterConfig.from_weights_file(f)


# ── conflicts ────────────────────────────────────────────────────
def test_conflict_kinds():
    bounds = [(0, 40), (40, 80)]
    assert conflict(cand(0, 5), cand(3, 8), bounds) == "overlap"
    assert conflict(cand(0, 5), cand(5, 9), bounds) == "adjacent"
    assert conflict(cand(0, 5, ttype="voice"), cand(10, 30, ttype="word_order"), bounds) == "structural"
    assert conflict(cand(0, 5, ttype="voice"), cand(45, 60, ttype="word_order"), bounds) is None
    a = cand(0, 5, ttype="synonym", original="big", replacement="large")
    b = cand(20, 25, ttype="repetition", original="huge", replacement="large")
    assert conflict(a, b, bounds) == "echo"


def test_select_rejects_echo_conflict():
    cfg = RewriterConfig(deterministic=True)
    a = cand(0, 5, "a", ttype="synonym", original="big", replacement="large", usefulness=1.0)
    b = cand(20, 25, "b", ttype="repetition", original="huge", replacement="large")
    for c in (a, b):
        score(c, cfg)
    selected, rejected = select([a, b], cfg, 5, random.Random(0))
    assert [c.rule_id for c in selected] == ["a"]
    assert "repeat" in rejected[0][1]


def test_engine_reproducible_with_seed():
    from humanizer.rule_engine import RuleEngine
    text = ("In order to finish, we basically need to completely eliminate the end result. "
            "The plan was reviewed by the team. The team really liked the plan.")
    eng = RuleEngine(RewriterConfig(intensity=Intensity.HEAVY))
    assert eng.run(text, seed=5).text == eng.run(text, seed=5).text
    applied = [r for r in eng.run(text, seed=5).history.records if r.accepted]
    assert applied and all(r.selection_probability is not None for r in applied)
