"""Stage 10: review session, feedback store and local web API."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from humanizer.analyzer import is_available
from humanizer.config import RewriterConfig
from humanizer.feedback import FeedbackRecord, FeedbackStore, assign_split, summarize
from humanizer.rewriter import RewriterPipeline
from humanizer.schemas import RewriteRequest
from humanizer.session import ReviewSession, word_diff
from humanizer.web.server import ReviewApp, make_server

needs_spacy = pytest.mark.skipif(not is_available(), reason="spaCy model not installed")
TEXT = "We met in order to discuss the end result. The plan was reviewed by the team."


def rules_session(validator=None):
    cfg = RewriterConfig(rules_only=True, deterministic=True)
    result = RewriterPipeline(cfg, backend=object()).rewrite_text(RewriteRequest(text=TEXT, seed=1))
    return ReviewSession.from_result(result, {"mode": "rules", "intensity": "moderate"}, validator)


# ── diff ─────────────────────────────────────────────────────────
def test_word_diff_round_trips():
    a, b = "We met in order to talk.", "We met to talk today."
    d = word_diff(a, b)
    assert "".join(s["text"] for s in d if s["op"] != "insert") == a
    assert "".join(s["text"] for s in d if s["op"] != "delete") == b


# ── session ──────────────────────────────────────────────────────
@needs_spacy
def test_reject_undo_redo_edit():
    s = rules_session()
    applied = [e for e in s.history.records if e.accepted]
    assert applied, "rules should change this text"
    full = s.rules_text
    target = next(e for e in applied if e.original_span == "end result")
    after_reject = s.decide(target.edit_id, "reject")
    assert "end result" in after_reject and after_reject != full
    assert s.undo() and s.rules_text == full
    assert s.redo() and s.rules_text == after_reject
    assert not s.redo()


@needs_spacy
def test_cannot_toggle_engine_skipped_edit():
    s = rules_session()
    skipped = [e for e in s.history.records if not e.accepted]
    if not skipped:
        pytest.skip("no skipped edits for this text")
    with pytest.raises(ValueError):
        s.decide(skipped[0].edit_id, "accept")


@needs_spacy
def test_unknown_edit_raises():
    with pytest.raises(KeyError):
        rules_session().decide("nope", "reject")


@needs_spacy
def test_session_view_has_everything_ui_needs():
    d = rules_session().to_dict()
    for k in ("original", "rules_text", "model_text", "edits", "chunks", "diff_rules", "diff_model", "can_undo"):
        assert k in d
    e = d["edits"][0]
    assert {"label", "from", "to", "reason", "confidence", "risk", "user", "applied"} <= set(e)


@needs_spacy
def test_toggling_revalidates_rules_version():
    calls = []

    class V:
        def validate(self, o, t):
            calls.append(t)
            from humanizer.schemas import ValidationReport
            return ValidationReport()

    s = rules_session(V())
    n = len(calls)
    s.decide(next(e.edit_id for e in s.history.records if e.accepted), "reject")
    assert len(calls) == n + 1


# ── feedback ─────────────────────────────────────────────────────
@needs_spacy
def test_feedback_records_edit_decisions(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    s = rules_session()
    first = next(e for e in s.history.records if e.accepted)
    s.decide(first.edit_id, "reject")
    rec = s.feedback(store, verdict="accept", final_text="Edited by hand.", rating=4, comment="ok")
    saved = store.all()
    assert len(saved) == 1 and saved[0]["id"] == rec.id
    assert saved[0]["final_text"] == "Edited by hand." and saved[0]["rating"] == 4
    assert any(e["decision"] == "rejected" and e["edit_id"] == first.edit_id for e in saved[0]["edits"])
    assert saved[0]["split"] in ("train", "dev", "eval")


def test_rejected_feedback_stores_no_final_text(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.add(FeedbackRecord("o", "p", None, "reject"))
    assert store.all()[0]["final_text"] is None


def test_feedback_validation():
    with pytest.raises(ValueError):
        FeedbackRecord("o", "p", "f", "maybe")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FeedbackRecord("o", "p", "f", "accept", rating=9)


def test_damaged_line_skipped(tmp_path):
    f = tmp_path / "fb.jsonl"
    f.write_text('{"verdict":"accept","edits":[]}\nnot json\n{"verdict":"reject","edits":[]}\n')
    assert len(FeedbackStore(f).all()) == 2


def test_summary_acceptance_rates():
    recs = [
        {"verdict": "accept", "rating": 4, "edits": [{"transform_type": "voice", "decision": "accepted"},
                                                      {"transform_type": "voice", "decision": "rejected"},
                                                      {"transform_type": "synonym", "decision": "untouched"}]},
        {"verdict": "reject", "edits": [{"transform_type": "voice", "decision": "rejected"}]},
    ]
    s = summarize(recs)
    assert s["transform_types"]["voice"]["acceptance_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert "synonym" not in s["transform_types"]  # untouched edits are not evidence
    assert s["verdicts"] == {"accept": 1, "reject": 1} and s["mean_rating"] == 4


def test_split_is_stable_per_text():
    assert assign_split("Same text. ") == assign_split("Same text.")


# ── HTTP API ─────────────────────────────────────────────────────
@pytest.fixture
def server(tmp_path):
    def factory(cfg):
        return RewriterPipeline(cfg.model_copy(update={"deterministic": True}), backend=object())

    app = ReviewApp(RewriterConfig(), feedback_path=tmp_path / "fb.jsonl", pipeline_factory=factory,
                    health=lambda: {"ollama": False, "model": "test", "analyzer": True})
    httpd = make_server(app, "127.0.0.1", 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", app
    httpd.shutdown()
    httpd.server_close()


def call(base, path, body=None):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_static_page_served(server):
    base, _ = server
    with urllib.request.urlopen(base + "/") as r:
        assert b"Rewriter" in r.read()
    assert call(base, "/../pyproject.toml")[0] == 404


def test_api_validation_errors(server):
    base, _ = server
    assert call(base, "/api/rewrite", {"text": "   "})[0] == 400
    assert call(base, "/api/rewrite", {"text": "x", "mode": "bogus"})[0] == 400
    assert call(base, "/api/rewrite", {"text": "x", "seed": "abc"})[0] == 400
    assert call(base, "/api/session/0123456789ab")[0] == 404


@needs_spacy
def test_api_full_review_flow(server):
    base, app = server
    status, s = call(base, "/api/rewrite", {"text": TEXT, "mode": "rules", "seed": 1})
    assert status == 200 and s["rules_text"] != s["original"]
    edit = next(e for e in s["edits"] if e["applied"])
    status, s2 = call(base, f"/api/session/{s['id']}/edit", {"edit_id": edit["id"], "decision": "reject"})
    assert status == 200 and s2["can_undo"] and s2["rules_text"] != s["rules_text"]
    status, s3 = call(base, f"/api/session/{s['id']}/undo", {})
    assert s3["rules_text"] == s["rules_text"]
    status, fb = call(base, f"/api/session/{s['id']}/feedback", {"verdict": "accept", "final_text": s3["rules_text"], "rating": 5})
    assert status == 200 and fb["saved"]
    status, summary = call(base, "/api/feedback/summary")
    assert summary["records"] == 1 and summary["verdicts"] == {"accept": 1}
    assert call(base, f"/api/session/{s['id']}/feedback", {"verdict": "nope"})[0] == 400
