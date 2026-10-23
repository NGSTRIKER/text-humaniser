"""Learning from review feedback (Stage 11, plan §11 — optional).

Turns the accept/reject decisions saved by the review UI (Stage 10,
``data/feedback.jsonl``) into per-transform-type weights that the
selection step already understands (``RewriterConfig.type_weights``).

Algorithm (deliberately simple and explainable):

* Count accepted vs rejected edits per transform type on the **train**
  split only.  ``untouched`` edits carry no signal and are ignored.
* Smooth each acceptance rate with a Beta(prior, prior) prior so a type
  with three reviews cannot swing to an extreme.
* Map the rate to a weight: 50 % acceptance → 1.0, linearly to
  ``min_weight`` at 0 % and ``max_weight`` at 100 %.
* Types with fewer than ``min_examples`` reviews keep their current weight.
* **Guard:** the new weights are checked on the **dev** split.  If they
  predict the reviewer's decisions worse than the current weights, the
  file is not written.

Rules that reviewers reject most of the time are listed separately so a
person can fix or disable them — they are not silently removed.

Fine-tuning a model is *not* done here; :func:`fine_tune_readiness`
only reports whether enough clean evaluation data exists to consider it.

Run: ``rewriter-learn --feedback data/feedback.jsonl --out learned_weights.json``
then ``rewriter --weights learned_weights.json ...``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from humanizer.feedback import FeedbackStore

# Fine-tuning needs far more than weight tuning; these are conservative floors.
FINE_TUNE_MIN_RECORDS = 1000
FINE_TUNE_MIN_EVAL = 200
FINE_TUNE_MIN_ACCEPTED = 500


@dataclass
class TypeStats:
    accepted: int = 0
    rejected: int = 0

    @property
    def total(self) -> int:
        return self.accepted + self.rejected


@dataclass
class LearnResult:
    weights: dict[str, float]
    stats: dict[str, dict[str, Any]]
    weak_rules: list[dict[str, Any]]
    dev_score_before: Optional[float]
    dev_score_after: Optional[float]
    applied: bool
    reason: str
    notes: list[str] = field(default_factory=list)


def _decisions(records: Iterable[dict], split: str, key: str) -> dict[str, TypeStats]:
    out: dict[str, TypeStats] = defaultdict(TypeStats)
    for rec in records:
        if rec.get("split", "train") != split:
            continue
        for e in rec.get("edits", []):
            d = e.get("decision")
            name = e.get(key) or "unknown"
            if d == "accepted":
                out[name].accepted += 1
            elif d == "rejected":
                out[name].rejected += 1
    return dict(out)


def smoothed_rate(s: TypeStats, prior: float = 2.0) -> float:
    """Beta-smoothed acceptance rate (pulls small samples towards 0.5)."""
    return (s.accepted + prior) / (s.total + 2 * prior)


def rate_to_weight(rate: float, min_weight: float = 0.2, max_weight: float = 1.6) -> float:
    if rate >= 0.5:
        w = 1.0 + (rate - 0.5) / 0.5 * (max_weight - 1.0)
    else:
        w = min_weight + rate / 0.5 * (1.0 - min_weight)
    return round(w, 3)


def dev_log_loss(dev: dict[str, TypeStats], weights: dict[str, float]) -> Optional[float]:
    """How well weights predict dev decisions (lower is better).

    A weight w is read as acceptance odds w : 1, i.e. p = w / (1 + w);
    a neutral weight of 1.0 predicts 50 %."""
    n, loss = 0, 0.0
    for t, s in dev.items():
        w = weights.get(t, 1.0)
        p = min(max(w / (1.0 + w), 1e-3), 1 - 1e-3)
        loss -= s.accepted * math.log(p) + s.rejected * math.log(1 - p)
        n += s.total
    return round(loss / n, 4) if n else None


def learn_weights(
    records: list[dict],
    current: Optional[dict[str, float]] = None,
    *,
    min_examples: int = 5,
    prior: float = 2.0,
    min_weight: float = 0.2,
    max_weight: float = 1.6,
    weak_rule_rate: float = 0.25,
) -> LearnResult:
    """Learn ``type_weights`` from feedback records (see module docs)."""
    if min_examples < 1 or prior <= 0 or not 0 < min_weight <= 1.0 <= max_weight:
        raise ValueError("invalid learning settings")
    current = dict(current or {})
    train = _decisions(records, "train", "transform_type")
    dev = _decisions(records, "dev", "transform_type")

    weights = dict(current)
    stats: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    for t, s in sorted(train.items()):
        rate = smoothed_rate(s, prior)
        enough = s.total >= min_examples
        if enough:
            weights[t] = rate_to_weight(rate, min_weight, max_weight)
        else:
            notes.append(f"{t}: only {s.total} reviews (need {min_examples}); weight unchanged")
        stats[t] = {"accepted": s.accepted, "rejected": s.rejected, "smoothed_rate": round(rate, 3),
                    "weight": weights.get(t, 1.0), "updated": enough}

    weak = []
    for rule, s in sorted(_decisions(records, "train", "rule_id").items()):
        r = smoothed_rate(s, prior)
        if s.total >= min_examples and r <= weak_rule_rate:
            weak.append({"rule_id": rule, "accepted": s.accepted, "rejected": s.rejected, "smoothed_rate": round(r, 3)})

    if not train:
        return LearnResult(current, {}, [], None, None, False, "no train-split edit decisions in feedback", notes)

    before, after = dev_log_loss(dev, current), dev_log_loss(dev, weights)
    if weights == current:
        applied, reason = False, "no type had enough reviews to change"
    elif before is None:
        applied, reason = True, "no dev-split feedback to check against; applied on train evidence only"
    elif after <= before:
        applied, reason = True, f"dev log-loss improved {before} -> {after}"
    else:
        applied, reason = False, f"dev log-loss got worse {before} -> {after}; kept current weights"
    return LearnResult(weights if applied else current, stats, weak, before, after, applied, reason, notes)


def fine_tune_readiness(records: list[dict]) -> dict[str, Any]:
    """Report whether the feedback set is big enough to *consider* fine-tuning."""
    accepted = [r for r in records if r.get("verdict") == "accept" and (r.get("final_text") or r.get("proposed_text"))]
    eval_n = sum(r.get("split") == "eval" for r in records)
    checks = {
        f"records >= {FINE_TUNE_MIN_RECORDS}": len(records) >= FINE_TUNE_MIN_RECORDS,
        f"eval-split records >= {FINE_TUNE_MIN_EVAL}": eval_n >= FINE_TUNE_MIN_EVAL,
        f"accepted rewrites >= {FINE_TUNE_MIN_ACCEPTED}": len(accepted) >= FINE_TUNE_MIN_ACCEPTED,
    }
    return {"ready": all(checks.values()), "records": len(records), "eval_records": eval_n,
            "accepted_rewrites": len(accepted), "checks": checks}


def format_result(res: LearnResult, readiness: dict[str, Any]) -> str:
    lines = ["== Learned transform weights =="]
    for t, s in res.stats.items():
        mark = "" if s["updated"] else "  (too few reviews)"
        lines.append(f"{t:<20} +{s['accepted']:<4} -{s['rejected']:<4} rate={s['smoothed_rate']:.2f} weight={s['weight']}{mark}")
    lines.append(f"\n{'APPLIED' if res.applied else 'NOT APPLIED'}: {res.reason}")
    if res.weak_rules:
        lines.append("\nRules reviewers mostly reject (consider fixing or disabling):")
        lines += [f"  {w['rule_id']}: +{w['accepted']} -{w['rejected']}" for w in res.weak_rules]
    lines.append(f"\nFine-tuning ready: {'yes' if readiness['ready'] else 'no'}")
    lines += [f"  [{'x' if ok else ' '}] {name}" for name, ok in readiness["checks"].items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rewriter-learn", description="Stage 11: learn selection weights from feedback.")
    ap.add_argument("--feedback", type=Path, default=Path("data/feedback.jsonl"))
    ap.add_argument("--weights", type=Path, help="Existing weights file to start from (and keep other settings).")
    ap.add_argument("--out", type=Path, default=Path("learned_weights.json"))
    ap.add_argument("--min-examples", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="Show results without writing a file.")
    args = ap.parse_args(argv)

    if not args.feedback.exists():
        print(f"No feedback file at {args.feedback}. Review some rewrites in rewriter-ui first.")
        return 1
    records = FeedbackStore(args.feedback).all()
    base: dict[str, Any] = {}
    if args.weights:
        try:
            base = json.loads(args.weights.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Cannot read {args.weights}: {exc}")
            return 1
    try:
        res = learn_weights(records, base.get("type_weights"), min_examples=args.min_examples)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    print(format_result(res, fine_tune_readiness(records)))
    if res.applied and not args.dry_run:
        args.out.write_text(json.dumps({**base, "type_weights": res.weights}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}; use it with: rewriter --weights {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
