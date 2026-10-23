"""End-to-end evaluation (Stage 9, plan §9).

Three parts:

1. **Mode comparison** — run every test paragraph through ``rules``
   (rule engine only), ``llm`` (model only, no rule suggestions) and
   ``hybrid`` (rules + structured model call) and score each output.
2. **Metrics** — meaning preservation comes from the Stage 8 validator
   (score, accept rate, critical issues, number/negation survival);
   naturalness uses cheap, transparent text statistics (sentence-length
   variation, repeated content words, filler phrases, wordiness change).
   These are proxies, not a human judgement.
3. **Threshold tuning** — a labelled set of good and bad paraphrases
   (``data/eval_pairs.jsonl``) is validated at every strictness level;
   the level with the best F1 for catching bad rewrites is recommended.

Run:  ``python -m humanizer.evaluation --modes rules`` (no Ollama needed)
or    ``rewriter-eval --modes rules llm hybrid --out report.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from humanizer.config import RewriterConfig
from humanizer.logging_config import get_logger
from humanizer.schemas import RewriteRequest

log = get_logger("evaluation")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODES = ("rules", "llm", "hybrid")
STRICTNESS_LEVELS = ("low", "medium", "high")

FILLERS = (
    "in order to", "due to the fact that", "it is important to note that", "at this point in time",
    "the utilization of", "a large number of", "in the event that", "for the purpose of",
    "it should be noted that", "basically", "actually", "very", "really",
)
STOPWORDS = set(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its "
    "this that these those which who whom what can could will would should may might must not no do "
    "does did has have had their they them he she his her we our you your i".split()
)
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENT = re.compile(r"(?<=[.!?])\s+")
_NUM = re.compile(r"\d[\d,.:/%-]*")
_NEG = re.compile(r"\b(not|no|never|none|nobody|nothing|neither|nor|without|cannot)\b|n't\b", re.I)


# ── Data ─────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Naturalness proxies ──────────────────────────────────────────

@dataclass
class TextStats:
    words: int
    sentences: int
    mean_sentence_len: float
    sentence_len_cv: float      # variation; very uniform lengths read as mechanical
    repeated_content_ratio: float
    filler_count: int
    type_token_ratio: float


def text_stats(text: str) -> TextStats:
    sents = [s for s in _SENT.split(text.strip()) if s]
    lens = [len(_WORD.findall(s)) for s in sents] or [0]
    words = [w.lower() for w in _WORD.findall(text)]
    content = [w for w in words if w not in STOPWORDS and len(w) > 2]
    mean = statistics.fmean(lens) if lens else 0.0
    cv = statistics.pstdev(lens) / mean if mean and len(lens) > 1 else 0.0
    repeated = 1 - len(set(content)) / len(content) if content else 0.0
    low = " " + text.lower() + " "
    fillers = sum(low.count(f" {f} ") for f in FILLERS)
    return TextStats(
        words=len(words), sentences=len(sents), mean_sentence_len=round(mean, 2),
        sentence_len_cv=round(cv, 3), repeated_content_ratio=round(repeated, 3),
        filler_count=fillers, type_token_ratio=round(len(set(words)) / len(words), 3) if words else 0.0,
    )


def naturalness_score(original: str, rewritten: str) -> float:
    """0..1 proxy. Rewards fewer fillers, less repetition, some length
    variation and not bloating the text. Unchanged text scores 0.5."""
    a, b = text_stats(original), text_stats(rewritten)
    score = 0.5
    score += 0.08 * max(-2, min(2, a.filler_count - b.filler_count))
    score += 0.6 * max(-0.25, min(0.25, a.repeated_content_ratio - b.repeated_content_ratio))
    score += 0.3 * max(-0.3, min(0.3, b.sentence_len_cv - a.sentence_len_cv))
    if a.words:
        growth = (b.words - a.words) / a.words
        score -= 0.5 * max(0.0, growth - 0.1)  # padding is unnatural
    return round(max(0.0, min(1.0, score)), 3)


def change_ratio(original: str, rewritten: str) -> float:
    """Share of words changed (word-level diff)."""
    import difflib

    a, b = original.split(), rewritten.split()
    if not a and not b:
        return 0.0
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    same = sum(m.size for m in sm.get_matching_blocks())
    return round(1 - 2 * same / (len(a) + len(b)), 3)


def numbers_preserved(original: str, rewritten: str) -> bool:
    norm = lambda t: sorted(n.strip(".,") for n in _NUM.findall(t))
    return norm(original) == norm(rewritten)


def negation_count_kept(original: str, rewritten: str) -> bool:
    return len(_NEG.findall(original)) == len(_NEG.findall(rewritten))


# ── Mode comparison ──────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    category: str
    mode: str
    output: str = ""
    ok: bool = True
    error: Optional[str] = None
    seconds: float = 0.0
    changed: bool = False
    change_ratio: float = 0.0
    validator_score: float = 0.0
    validator_decision: str = ""
    critical_issues: int = 0
    numbers_preserved: bool = True
    negation_kept: bool = True
    naturalness: float = 0.5


@dataclass
class ModeSummary:
    mode: str
    cases: int
    failures: int
    accept_rate: float
    mean_validator_score: float
    critical_issue_cases: int
    numbers_preserved_rate: float
    negation_kept_rate: float
    mean_naturalness: float
    mean_change_ratio: float
    unchanged_rate: float
    mean_seconds: float


def config_for_mode(mode: str, base: RewriterConfig) -> RewriterConfig:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    upd: dict[str, Any] = {
        "rules": {"rules_enabled": True, "rules_only": True},
        "llm": {"rules_enabled": False, "rules_only": False, "llm_mode": "plain"},
        "hybrid": {"rules_enabled": True, "rules_only": False, "llm_mode": "structured"},
    }[mode]
    return base.model_copy(update=upd)


def run_case(case: dict, mode: str, pipeline, validator, intensity: str, seed: int) -> CaseResult:
    r = CaseResult(case_id=case["id"], category=case.get("category", ""), mode=mode)
    src = case["text"]
    t0 = time.monotonic()
    try:
        res = pipeline.rewrite_text(RewriteRequest(text=src, intensity=intensity, seed=seed))
        out = res.rewritten_text
        if getattr(res.status, "value", res.status) == "failed":
            r.ok, r.error = False, "pipeline reported failure (model unavailable?)"
    except Exception as exc:  # one broken case must not stop the run
        out, r.ok, r.error = src, False, f"{type(exc).__name__}: {exc}"
    r.seconds = round(time.monotonic() - t0, 3)
    r.output = out
    r.changed = out.strip() != src.strip()
    r.change_ratio = change_ratio(src, out)
    rep = validator.validate(src, out)
    r.validator_score, r.validator_decision = rep.overall_score, rep.decision
    r.critical_issues = len(rep.errors)
    r.numbers_preserved = numbers_preserved(src, out)
    r.negation_kept = negation_count_kept(src, out)
    r.naturalness = naturalness_score(src, out)
    return r


def summarize(mode: str, results: list[CaseResult]) -> ModeSummary:
    n = len(results) or 1
    mean = lambda xs: round(statistics.fmean(xs), 3) if xs else 0.0
    return ModeSummary(
        mode=mode, cases=len(results), failures=sum(not r.ok for r in results),
        accept_rate=round(sum(r.validator_decision == "accept" for r in results) / n, 3),
        mean_validator_score=mean([r.validator_score for r in results]),
        critical_issue_cases=sum(r.critical_issues > 0 for r in results),
        numbers_preserved_rate=round(sum(r.numbers_preserved for r in results) / n, 3),
        negation_kept_rate=round(sum(r.negation_kept for r in results) / n, 3),
        mean_naturalness=mean([r.naturalness for r in results]),
        mean_change_ratio=mean([r.change_ratio for r in results]),
        unchanged_rate=round(sum(not r.changed for r in results) / n, 3),
        mean_seconds=mean([r.seconds for r in results]),
    )


def compare_modes(
    cases: list[dict],
    modes: Iterable[str] = MODES,
    *,
    config: Optional[RewriterConfig] = None,
    backend: object | None = None,
    intensity: str = "moderate",
    seed: int = 42,
) -> dict[str, Any]:
    from humanizer.rewriter import RewriterPipeline
    from humanizer.validation import Validator

    base = config or RewriterConfig()
    validator = Validator(base)
    per_mode: dict[str, list[CaseResult]] = {}
    skipped: dict[str, str] = {}
    for mode in modes:
        cfg = config_for_mode(mode, base)
        be = backend
        if mode != "rules" and be is None:
            from humanizer.llm_backend import OllamaBackend

            be = OllamaBackend(cfg)
            if not be.health_check():
                skipped[mode] = "Ollama not reachable — start it to evaluate this mode"
                log.warning("Skipping %s mode: %s", mode, skipped[mode])
                continue
        pipeline = RewriterPipeline(cfg, backend=be or _NoBackend())
        per_mode[mode] = [run_case(c, mode, pipeline, validator, intensity, seed) for c in cases]
    return {
        "summary": {m: asdict(summarize(m, rs)) for m, rs in per_mode.items()},
        "skipped_modes": skipped,
        "by_case": {m: [asdict(r) for r in rs] for m, rs in per_mode.items()},
    }


class _NoBackend:
    """Placeholder for rules-only mode; the pipeline never calls it."""

    def health_check(self) -> bool:
        return False

    def generate(self, *a, **k) -> str:  # pragma: no cover - never reached
        raise RuntimeError("no LLM backend in rules mode")


# ── Threshold tuning ─────────────────────────────────────────────

@dataclass
class ThresholdResult:
    strictness: str
    true_rejects: int      # bad rewrite rejected (good)
    false_rejects: int     # good rewrite rejected
    missed_bad: int        # bad rewrite accepted (dangerous)
    true_accepts: int
    precision: float
    recall: float
    f1: float
    per_pair: list[dict] = field(default_factory=list)


def evaluate_threshold(pairs: list[dict], strictness: str, config: Optional[RewriterConfig] = None) -> ThresholdResult:
    from humanizer.validation import Validator

    cfg = (config or RewriterConfig()).model_copy(update={"semantic_strictness": strictness})
    v = Validator(cfg)
    tp = fp = fn = tn = 0
    rows = []
    for p in pairs:
        rep = v.validate(p["original"], p["rewritten"])
        rejected = not rep.is_valid
        bad = p["label"] == "bad"
        tp += bad and rejected
        fn += bad and not rejected
        fp += (not bad) and rejected
        tn += (not bad) and not rejected
        rows.append({"id": p["id"], "label": p["label"], "rejected": rejected, "score": rep.overall_score})
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return ThresholdResult(strictness, tp, fp, fn, tn, round(prec, 3), round(rec, 3), round(f1, 3), rows)


def tune_thresholds(pairs: list[dict], config: Optional[RewriterConfig] = None) -> dict[str, Any]:
    """Pick the strictness that best separates good from bad rewrites.

    Missing a meaning change is worse than an unnecessary rejection, so ties
    on F1 go to higher recall, then to the stricter level."""
    results = [evaluate_threshold(pairs, s, config) for s in STRICTNESS_LEVELS]
    order = {s: i for i, s in enumerate(STRICTNESS_LEVELS)}
    best = max(results, key=lambda r: (r.f1, r.recall, order[r.strictness]))
    return {
        "recommended_strictness": best.strictness,
        "current_strictness": (config or RewriterConfig()).semantic_strictness,
        "levels": [asdict(r) for r in results],
        "missed_bad_ids": [row["id"] for row in best.per_pair if row["label"] == "bad" and not row["rejected"]],
        "false_reject_ids": [row["id"] for row in best.per_pair if row["label"] == "good" and row["rejected"]],
    }


# ── Report ───────────────────────────────────────────────────────

def format_report(report: dict[str, Any]) -> str:
    lines = ["== Mode comparison =="]
    cols = ("accept_rate", "mean_validator_score", "critical_issue_cases", "numbers_preserved_rate",
            "negation_kept_rate", "mean_naturalness", "mean_change_ratio", "unchanged_rate", "mean_seconds")
    summary = report.get("modes", {}).get("summary", {})
    if summary:
        lines.append(f"{'metric':<24}" + "".join(f"{m:>10}" for m in summary))
        for c in cols:
            lines.append(f"{c:<24}" + "".join(f"{summary[m][c]:>10}" for m in summary))
    for m, why in report.get("modes", {}).get("skipped_modes", {}).items():
        lines.append(f"(skipped {m}: {why})")
    t = report.get("thresholds")
    if t:
        lines += ["", "== Validator threshold tuning =="]
        for lv in t["levels"]:
            lines.append(f"{lv['strictness']:<7} precision={lv['precision']:.2f} recall={lv['recall']:.2f} "
                         f"f1={lv['f1']:.2f} missed_bad={lv['missed_bad']} false_rejects={lv['false_rejects']}")
        lines.append(f"recommended semantic_strictness: {t['recommended_strictness']} "
                     f"(current: {t['current_strictness']})")
        if t["missed_bad_ids"]:
            lines.append("still missed: " + ", ".join(t["missed_bad_ids"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rewriter-eval", description="Stage 9 end-to-end evaluation.")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--cases", type=Path, default=DATA_DIR / "test_paragraphs.jsonl")
    ap.add_argument("--pairs", type=Path, default=DATA_DIR / "eval_pairs.jsonl")
    ap.add_argument("--intensity", default="moderate", choices=["light", "moderate", "heavy"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-tuning", action="store_true", help="Skip validator threshold tuning.")
    ap.add_argument("--out", type=Path, help="Write the full JSON report here.")
    args = ap.parse_args(argv)

    report: dict[str, Any] = {
        "modes": compare_modes(load_jsonl(args.cases), args.modes, intensity=args.intensity, seed=args.seed)
    }
    if not args.no_tuning:
        report["thresholds"] = tune_thresholds(load_jsonl(args.pairs))
    print(format_report(report))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
