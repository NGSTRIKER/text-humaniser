"""Stage 11 feedback learning tests."""

import json

import pytest

from humanizer import learning as lr
from humanizer.config import RewriterConfig


def rec(split, edits, verdict="partial"):
    return {"split": split, "verdict": verdict, "original_text": "x", "proposed_text": "y",
            "edits": [{"transform_type": t, "rule_id": r, "decision": d} for t, r, d in edits]}


def many(split, t, rule, acc, rej):
    return [rec(split, [(t, rule, "accepted")])] * acc + [rec(split, [(t, rule, "rejected")])] * rej


def test_rate_to_weight_mapping():
    assert lr.rate_to_weight(0.5) == 1.0
    assert lr.rate_to_weight(0.0) == 0.2
    assert lr.rate_to_weight(1.0) == 1.6


def test_smoothing_pulls_small_samples_to_half():
    assert lr.smoothed_rate(lr.TypeStats(1, 0)) == pytest.approx(0.6)


def test_rejected_type_goes_down_and_liked_type_goes_up():
    data = many("train", "synonym", "synonym.repeat", 2, 18) + many("train", "filler", "filler.cut", 18, 2)
    res = lr.learn_weights(data)
    assert res.applied
    assert res.weights["synonym"] < 0.6
    assert res.weights["filler"] > 1.3
    assert [w["rule_id"] for w in res.weak_rules] == ["synonym.repeat"]


def test_few_reviews_leave_weight_unchanged():
    res = lr.learn_weights(many("train", "voice", "v", 0, 3), {"voice": 0.8})
    assert res.weights == {"voice": 0.8} and not res.applied


def test_untouched_and_other_splits_are_ignored():
    data = [rec("train", [("voice", "v", "untouched")])] * 10 + many("eval", "voice", "v", 0, 10)
    assert lr.learn_weights(data).reason.startswith("no train-split")


def test_dev_guard_blocks_weights_that_predict_worse():
    data = many("train", "synonym", "s", 0, 20) + many("dev", "synonym", "s", 20, 0)
    res = lr.learn_weights(data)
    assert not res.applied and res.weights == {}
    assert "worse" in res.reason


def test_dev_guard_allows_consistent_weights():
    data = many("train", "synonym", "s", 1, 19) + many("dev", "synonym", "s", 1, 9)
    res = lr.learn_weights(data)
    assert res.applied and res.dev_score_after < res.dev_score_before


def test_invalid_settings():
    with pytest.raises(ValueError):
        lr.learn_weights([], min_examples=0)


def test_fine_tune_readiness_small_set_not_ready():
    r = lr.fine_tune_readiness(many("train", "x", "x", 5, 5))
    assert r["ready"] is False and r["records"] == 10


def test_cli_writes_file_loadable_by_config(tmp_path, capsys):
    fb = tmp_path / "fb.jsonl"
    fb.write_text("\n".join(json.dumps(r) for r in many("train", "synonym", "s", 1, 19)))
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"weight_meaning_risk": 2.0}))
    out = tmp_path / "w.json"
    assert lr.main(["--feedback", str(fb), "--weights", str(base), "--out", str(out)]) == 0
    cfg = RewriterConfig.from_weights_file(out)
    assert cfg.type_weights["synonym"] < 0.6
    assert cfg.weight_meaning_risk == 2.0


def test_cli_missing_feedback(tmp_path):
    assert lr.main(["--feedback", str(tmp_path / "none.jsonl")]) == 1
