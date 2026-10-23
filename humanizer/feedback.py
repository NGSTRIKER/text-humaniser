"""Local feedback store (plan §16).

Feedback is appended to a JSONL file (one JSON object per line), never
sent anywhere.  Each record keeps the original, the proposed rewrite,
the final text the user accepted (if any), per-edit decisions, settings
and validation results.

Generated output is NOT treated as a correct example: only records with
``verdict == "accept"`` and no user rejection of the final text should
be considered for later learning (Stage 11), and :func:`summarize`
reports per-transform acceptance rates rather than raw counts.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

Verdict = Literal["accept", "reject", "partial"]
Split = Literal["train", "dev", "eval"]


@dataclass
class EditFeedback:
    edit_id: str
    rule_id: str
    transform_type: str
    original_span: str
    replacement: str
    decision: Literal["accepted", "rejected", "untouched"]


@dataclass
class FeedbackRecord:
    original_text: str
    proposed_text: str
    final_text: Optional[str]
    verdict: Verdict
    edits: list[EditFeedback] = field(default_factory=list)
    intensity: str = "moderate"
    tone: str = ""
    mode: str = "hybrid"
    seed: Optional[int] = None
    validation: dict[str, Any] = field(default_factory=dict)
    rating: Optional[int] = None       # 1–5 naturalness rating
    comment: str = ""
    split: Split = "train"             # keep train/dev/eval separate (§16)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.verdict not in ("accept", "reject", "partial"):
            raise ValueError(f"verdict must be accept/reject/partial, got {self.verdict!r}")
        if self.rating is not None and not 1 <= int(self.rating) <= 5:
            raise ValueError("rating must be between 1 and 5")
        if self.split not in ("train", "dev", "eval"):
            raise ValueError("split must be train/dev/eval")


def assign_split(text: str) -> Split:
    """Stable split from the text itself, so the same text never lands in two splits."""
    import hashlib
    h = int(hashlib.sha256(text.strip().encode()).hexdigest(), 16) % 10
    return "eval" if h == 0 else "dev" if h == 1 else "train"


class FeedbackStore:
    """Append-only JSONL feedback file (thread-safe within one process)."""

    def __init__(self, path: str | Path = "data/feedback.jsonl") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, record: FeedbackRecord) -> FeedbackRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return record

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a damaged line must not hide the rest

    def all(self) -> list[dict[str, Any]]:
        return list(self)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Acceptance rate per transform type and overall verdict counts."""
    per_type: dict[str, dict[str, int]] = defaultdict(lambda: {"accepted": 0, "rejected": 0})
    verdicts: dict[str, int] = defaultdict(int)
    ratings = []
    for r in records:
        verdicts[r.get("verdict", "?")] += 1
        if r.get("rating"):
            ratings.append(r["rating"])
        for e in r.get("edits", []):
            if e.get("decision") in ("accepted", "rejected"):
                per_type[e["transform_type"]][e["decision"]] += 1
    types = {
        t: {**c, "acceptance_rate": round(c["accepted"] / (c["accepted"] + c["rejected"]), 3)}
        for t, c in per_type.items() if c["accepted"] + c["rejected"]
    }
    return {
        "records": len(records),
        "verdicts": dict(verdicts),
        "mean_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "transform_types": types,
    }
