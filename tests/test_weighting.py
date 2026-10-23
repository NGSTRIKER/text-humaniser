"""Scoring, gating and selection tests (no spaCy needed)."""

from __future__ import annotations

import random

from humanizer.candidates import Candidate
from humanizer.config import Intensity, RewriterConfig
from humanizer.weighting import edit_cap, gate, score, select


def cand(start, end, **kw) -> Candidate:
    base = dict(rule_id="t", transform_type="wordiness", char_start=start, char_end=end,
                original="x" * (end - start), replacement="y", reason="r")
    base.update(kw)
    return Candidate(**base)


def test_score_formula_uses_configurable_weights():
    c = cand(0, 3, usefulness=1.0, context_fit=0.0, confidence=0.0,
             diversity_gain=0.0, meaning_risk=0.0, grammar_risk=0.0)
    assert score(c, RewriterConfig(weight_usefulness=2.0)) == 2.0
    c2 = cand(0, 3, usefulness=0.0, context_fit=0.0, confidence=0.0,
              diversity_gain=0.0, meaning_risk=1.0, grammar_risk=0.0)
    assert score(c2, RewriterConfig(weight_meaning_risk=3.0)) == -3.0


def test_high_risk_rejected_even_if_useful():
    c = cand(0, 3, meaning_risk=0.5, usefulness=1.0)
    assert "meaning risk" in gate(c, RewriterConfig(), Intensity.MODERATE, [])


def test_meaning_risk_limit_depends_on_intensity():
    c = cand(0, 3, meaning_risk=0.15)
    cfg = RewriterConfig()
    assert gate(c, cfg, Intensity.LIGHT, []) is not None
    assert gate(c, cfg, Intensity.MODERATE, []) is None


def test_grammar_failure_rejected():
    c = cand(0, 3, grammar_ok=False, note="bad")
    assert "grammar" in gate(c, RewriterConfig(), Intensity.HEAVY, [])


def test_flag_only_rejected():
    c = cand(0, 3, replacement=None)
    assert "flag only" in gate(c, RewriterConfig(), Intensity.HEAVY, [])


def test_protected_span_rejected():
    c = cand(0, 5)
    assert "protected" in gate(c, RewriterConfig(), Intensity.HEAVY, [(2, 4, "negation")])


def test_overlaps_resolved_deterministically():
    cfg = RewriterConfig(deterministic=True)
    a, b = cand(0, 5, rule_id="a"), cand(3, 8, rule_id="b", usefulness=1.0)
    for c in (a, b):
        score(c, cfg)
    selected, rejected = select([a, b], cfg, 10, random.Random(0))
    assert [c.rule_id for c in selected] == ["b"]
    assert "overlaps" in rejected[0][1]


def test_contained_edit_subsumed():
    cfg = RewriterConfig(deterministic=True)
    big, small = cand(0, 10, rule_id="big"), cand(2, 5, rule_id="small")
    for c in (big, small):
        score(c, cfg)
    selected, _ = select([big, small], cfg, 10, random.Random(0))
    assert [c.rule_id for c in selected] == ["big"]


def test_cap_limits_edits():
    cfg = RewriterConfig(deterministic=True)
    cands = [cand(i * 10, i * 10 + 3) for i in range(6)]
    for c in cands:
        score(c, cfg)
    selected, rejected = select(cands, cfg, 2, random.Random(0))
    assert len(selected) == 2
    assert sum("limit" in r for _, r in rejected) == 4


def test_same_seed_same_selection():
    cfg = RewriterConfig()
    def pick(seed):
        cands = [cand(i * 10, i * 10 + 3, rule_id=str(i), usefulness=i / 10) for i in range(8)]
        for c in cands:
            score(c, cfg)
        return [c.rule_id for c in select(cands, cfg, 3, random.Random(seed))[0]]
    assert pick(7) == pick(7)


def test_edit_cap_scales_with_length_and_intensity():
    assert edit_cap(100, Intensity.LIGHT) < edit_cap(100, Intensity.HEAVY)
    assert edit_cap(5, Intensity.LIGHT) == 1
