"""Review session: inspect, accept, reject and undo individual edits (plan §15, Stage 10).

Independent of any UI so it can be tested directly; the web interface
(:mod:`humanizer.web`) is a thin JSON layer over it.

Two versions are tracked:

* **rules version** — the original with the rule edits the user keeps.
  Toggling an edit re-renders it from the ORIGINAL (offsets never go stale)
  and re-validates it.
* **model version** — the LLM/hybrid output, shown for comparison.

The user picks either version (or edits the text by hand) and submits a
verdict, which is stored by :class:`~humanizer.feedback.FeedbackStore`.
"""

from __future__ import annotations

import difflib
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

from humanizer.edit_history import TYPE_LABELS, EditHistory
from humanizer.feedback import EditFeedback, FeedbackRecord, FeedbackStore, assign_split
from humanizer.schemas import RewriteResult, ValidationReport

Decision = Literal["accept", "reject", "reset"]
_TOKEN = re.compile(r"\s+|\w+(?:'\w+)?|[^\w\s]")


def word_diff(a: str, b: str) -> list[dict[str, str]]:
    """Word-level diff as ``[{"op": equal|delete|insert, "text": …}]``."""
    ta, tb = _TOKEN.findall(a), _TOKEN.findall(b)
    out: list[dict[str, str]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        if op == "equal":
            out.append({"op": "equal", "text": "".join(ta[i1:i2])})
        else:
            if i2 > i1:
                out.append({"op": "delete", "text": "".join(ta[i1:i2])})
            if j2 > j1:
                out.append({"op": "insert", "text": "".join(tb[j1:j2])})
    return out


def report_dict(rep: Optional[ValidationReport]) -> Optional[dict[str, Any]]:
    if rep is None:
        return None
    keys = ("is_valid", "decision", "overall_score", "confidence", "errors", "warnings",
            "suspected_meaning_changes", "missing_information", "unsupported_additions", "grammar_problems")
    return {k: getattr(rep, k) for k in keys}


@dataclass
class ReviewSession:
    original: str
    history: EditHistory
    model_text: str
    result: RewriteResult
    settings: dict[str, Any] = field(default_factory=dict)
    validator: Any = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _undo: list[tuple[str, Optional[bool]]] = field(default_factory=list)
    _redo: list[tuple[str, Optional[bool]]] = field(default_factory=list)
    _rules_validation: Optional[ValidationReport] = None
    submitted: bool = False

    # ── creation ─────────────────────────────────────────────────
    @classmethod
    def from_result(cls, result: RewriteResult, settings: dict[str, Any], validator: Any = None) -> "ReviewSession":
        original = result.original_text.strip()
        history = EditHistory(original_text=original, records=list(result.edits))
        s = cls(original, history, result.rewritten_text, result, settings, validator)
        s._revalidate()
        return s

    # ── edit decisions ───────────────────────────────────────────
    @property
    def rules_text(self) -> str:
        return self.history.render()

    def decide(self, edit_id: str, decision: Decision) -> str:
        rec = self.history.get(edit_id)
        if rec is None:
            raise KeyError(f"unknown edit {edit_id}")
        if not rec.accepted:
            raise ValueError(f"edit {edit_id} was not applied by the engine ({rec.rejection_reason})")
        new = {"accept": True, "reject": False, "reset": None}[decision]
        if rec.user_decision == new:
            return self.rules_text
        self._undo.append((edit_id, rec.user_decision))
        self._redo.clear()
        rec.user_decision = new
        self._revalidate()
        return self.rules_text

    def undo(self) -> bool:
        if not self._undo:
            return False
        edit_id, previous = self._undo.pop()
        rec = self.history.get(edit_id)
        if rec is None:
            return False
        self._redo.append((edit_id, rec.user_decision))
        rec.user_decision = previous
        self._revalidate()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        edit_id, value = self._redo.pop()
        rec = self.history.get(edit_id)
        if rec is None:
            return False
        self._undo.append((edit_id, rec.user_decision))
        rec.user_decision = value
        self._revalidate()
        return True

    def _revalidate(self) -> None:
        if self.validator is None or self.rules_text == self.original:
            self._rules_validation = None
            return
        try:
            self._rules_validation = self.validator.validate(self.original, self.rules_text)
        except Exception as exc:  # never break the UI over a check
            self._rules_validation = ValidationReport(is_valid=False, decision="reject", errors=[f"validation error: {exc}"])

    # ── feedback ─────────────────────────────────────────────────
    def feedback(self, store: FeedbackStore, *, verdict: str, final_text: Optional[str] = None,
                 rating: Optional[int] = None, comment: str = "") -> FeedbackRecord:
        edits = [
            EditFeedback(r.edit_id, r.rule_id, r.transform_type, r.original_span, r.replacement,
                         "accepted" if r.user_decision is True else "rejected" if r.user_decision is False else "untouched")
            for r in self.history.records if r.accepted
        ]
        val = self._final_validation(final_text)
        record = FeedbackRecord(
            original_text=self.original, proposed_text=self.model_text,
            final_text=final_text if verdict != "reject" else None, verdict=verdict, edits=edits,
            intensity=str(self.settings.get("intensity", "")), tone=str(self.settings.get("tone", "")),
            mode=str(self.settings.get("mode", "")), seed=self.settings.get("seed"),
            validation=val or {}, rating=rating, comment=comment[:2000], split=assign_split(self.original),
        )
        store.add(record)
        self.submitted = True
        return record

    def _final_validation(self, final_text: Optional[str]) -> Optional[dict[str, Any]]:
        if not final_text or self.validator is None:
            return None
        try:
            return report_dict(self.validator.validate(self.original, final_text))
        except Exception:
            return None

    # ── view model ───────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        edits = []
        for r in sorted(self.history.records, key=lambda r: (not r.accepted, r.char_start)):  # applied first
            edits.append({
                "id": r.edit_id, "label": TYPE_LABELS.get(r.transform_type, "Changed"),
                "type": r.transform_type, "rule": r.rule_id, "from": r.original_span, "to": r.replacement,
                "reason": r.reason, "confidence": round(r.confidence, 2), "risk": round(r.semantic_risk, 2),
                "probability": r.selection_probability, "applied": bool(r.accepted),
                "user": {True: "accepted", False: "rejected", None: "pending"}[r.user_decision],
                "skipped_because": r.rejection_reason,
            })
        chunks = []
        for cr in self.result.chunk_results:
            chunks.append({
                "index": cr.chunk.index, "source": cr.source, "success": cr.success, "error": cr.error,
                "uncertain": cr.uncertain, "model_rejected": cr.rejected_suggestions, "warnings": cr.warnings,
                "validation": report_dict(cr.validation),
            })
        return {
            "id": self.id,
            "original": self.original,
            "rules_text": self.rules_text,
            "model_text": self.model_text,
            "status": self.result.status.value,
            "settings": self.settings,
            "edits": edits,
            "chunks": chunks,
            "rules_validation": report_dict(self._rules_validation),
            "diff_rules": word_diff(self.original, self.rules_text),
            "diff_model": word_diff(self.original, self.model_text),
            "can_undo": bool(self._undo),
            "can_redo": bool(self._redo),
            "submitted": self.submitted,
        }


class SessionStore:
    """In-memory sessions with a small cap (oldest dropped first)."""

    def __init__(self, limit: int = 50) -> None:
        self.limit = limit
        self._items: dict[str, ReviewSession] = {}
        self._lock = threading.Lock()

    def add(self, s: ReviewSession) -> ReviewSession:
        with self._lock:
            self._items[s.id] = s
            while len(self._items) > self.limit:
                self._items.pop(next(iter(self._items)))
        return s

    def get(self, sid: str) -> ReviewSession:
        with self._lock:
            if sid not in self._items:
                raise KeyError(f"unknown session {sid}")
            return self._items[sid]
