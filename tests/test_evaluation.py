"""Stage 9 evaluation tests (no Ollama needed; a fake model is used)."""

import pytest

from humanizer import evaluation as ev
from humanizer.config import RewriterConfig

pytest.importorskip("spacy")


class EchoBackend:
    """Fake model that returns the text unchanged (plain mode)."""

    def health_check(self):
        return True

    def generate(self, prompt, system="", **kw):
        return prompt.strip().split("\n")[-1]


def test_change_ratio():
    assert ev.change_ratio("a b c d", "a b c d") == 0.0
    assert ev.change_ratio("a b c d", "w x y z") == 1.0


def test_numbers_and_negation_checks():
    assert ev.numbers_preserved("Rose 12% in 2023.", "In 2023 it rose 12%.")
    assert not ev.numbers_preserved("Rose 12%.", "Rose 21%.")
    assert ev.negation_count_kept("He did not go.", "He didn't go.")
    assert not ev.negation_count_kept("He did not go.", "He went.")


def test_naturalness_rewards_removing_filler_and_penalises_padding():
    src = "In order to win, it is important to note that we must train."
    assert ev.naturalness_score(src, src) == 0.5
    assert ev.naturalness_score(src, "To win, we must train.") > 0.5
    padded = src + " This is truly, really, very much a fact that everyone should always remember forever."
    assert ev.naturalness_score(src, padded) < 0.5


def test_config_for_mode():
    base = RewriterConfig()
    assert ev.config_for_mode("rules", base).rules_only is True
    llm = ev.config_for_mode("llm", base)
    assert llm.rules_enabled is False and llm.rules_only is False
    assert ev.config_for_mode("hybrid", base).llm_mode == "structured"
    with pytest.raises(ValueError):
        ev.config_for_mode("magic", base)


def test_compare_modes_rules_only_runs_all_cases():
    cases = ev.load_jsonl(ev.DATA_DIR / "test_paragraphs.jsonl")[:3]
    rep = ev.compare_modes(cases, ["rules"])
    s = rep["summary"]["rules"]
    assert s["cases"] == 3 and s["failures"] == 0
    assert 0.0 <= s["mean_validator_score"] <= 1.0


def test_compare_modes_with_fake_llm_reports_each_mode():
    cases = ev.load_jsonl(ev.DATA_DIR / "test_paragraphs.jsonl")[:2]
    rep = ev.compare_modes(cases, ["rules", "llm"], backend=EchoBackend())
    assert set(rep["summary"]) == {"rules", "llm"}
    assert len(rep["by_case"]["llm"]) == 2


def test_threshold_tuning_catches_known_meaning_changes():
    pairs = ev.load_jsonl(ev.DATA_DIR / "eval_pairs.jsonl")
    t = ev.tune_thresholds(pairs)
    assert t["recommended_strictness"] in ev.STRICTNESS_LEVELS
    best = next(lv for lv in t["levels"] if lv["strictness"] == t["recommended_strictness"])
    # The validator must catch most deliberately broken rewrites.
    assert best["recall"] >= 0.75
    for must in ("bad_negation", "bad_number"):
        assert must not in t["missed_bad_ids"]


def test_cli_rules_mode(tmp_path, capsys):
    out = tmp_path / "r.json"
    cases = tmp_path / "c.jsonl"
    cases.write_text((ev.DATA_DIR / "test_paragraphs.jsonl").read_text().splitlines()[0] + "\n")
    assert ev.main(["--modes", "rules", "--cases", str(cases), "--out", str(out)]) == 0
    assert out.exists()
    assert "recommended semantic_strictness" in capsys.readouterr().out
