"""Stage 7: structured LLM requests, output validation, retries, memory handling."""

from __future__ import annotations

import json

import httpx
import pytest

from humanizer.config import Intensity, RewriterConfig
from humanizer.llm_backend import LLMError, LLMMemoryError, OllamaBackend
from humanizer.llm_request import OUTPUT_SCHEMA, build_request, prompt_budget, suggestions_for_chunk
from humanizer.llm_response import OutputFormatError, parse_response
from humanizer.rewriter import RewriterPipeline
from humanizer.schemas import EditRecord, RewriteRequest, RewriteStatus


def reply(text, applied=(), rejected=(), uncertain=()):
    return json.dumps({"rewritten_text": text, "applied": list(applied),
                       "rejected": list(rejected), "uncertain": list(uncertain)})


class ScriptedBackend:
    """Returns queued replies and records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def health_check(self):
        return True

    def generate(self, prompt, system="", **kw):
        self.calls.append({"prompt": prompt, "system": system, **kw})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


CFG = RewriterConfig(rules_enabled=False)


# ── request building ─────────────────────────────────────────────
def test_request_contains_required_parts():
    req = build_request(
        "Alice paid 40 dollars.", CFG, intensity=Intensity.LIGHT,
        suggestions=[{"id": "e1", "type": "wordiness", "from": "a", "to": "b", "why": "w", "confidence": 0.9}],
        analysis_summary={"entities": ["Alice"], "numbers": ["40"]},
        protected_terms=["dollars"],
    )
    payload = json.loads(req.to_prompt())
    for key in ("text", "intensity", "suggestions", "protected", "analysis", "constraints",
                "max_output_chars", "output_schema", "tone"):
        assert key in payload
    assert set(payload["protected"]) == {"dollars", "Alice", "40"}
    assert payload["intensity"] == "light"
    assert payload["max_output_chars"] == int(len("Alice paid 40 dollars.") * 1.3) + 20


def test_request_trimmed_to_budget():
    cfg = RewriterConfig(context_window=600, max_tokens=100)
    sugg = [{"id": f"e{i}", "type": "t", "from": "x" * 40, "to": "y" * 40, "why": "z" * 40,
             "confidence": 1 - i / 100} for i in range(20)]
    req = build_request("Short text.", cfg, suggestions=sugg, analysis_summary={"entities": ["A"] * 50})
    assert cfg.estimate_tokens(req.to_prompt()) <= prompt_budget(cfg)
    assert req.analysis == {}
    assert req.suggestions and req.suggestions[0]["id"] == "e0"  # highest confidence kept


def test_oversized_chunk_rejected():
    with pytest.raises(ValueError, match="context window"):
        build_request("word " * 400, RewriterConfig(context_window=600, max_tokens=100))


def test_suggestions_only_inside_chunk_and_accepted():
    edits = [
        EditRecord(edit_id="e1", char_start=2, char_end=5, accepted=True, confidence=0.5),
        EditRecord(edit_id="e2", char_start=8, char_end=20, accepted=True),
        EditRecord(edit_id="e3", char_start=3, char_end=4, accepted=False),
        EditRecord(edit_id="e4", char_start=1, char_end=2, accepted=True, user_decision=False),
    ]
    assert [s["id"] for s in suggestions_for_chunk(edits, 0, 10)] == ["e1"]


# ── response validation ──────────────────────────────────────────
def _req(text="Bob has 3 cats.", protected=("Bob", "3")):
    r = build_request(text, CFG)
    r.protected = list(protected)
    return r


def test_parse_tolerates_fences_and_think_blocks():
    raw = "<think>hmm</think>\n```json\n" + reply("Bob owns 3 cats.") + "\n```"
    assert parse_response(raw, _req()).rewritten_text == "Bob owns 3 cats."


def test_parse_extracts_json_after_chatter():
    raw = "Sure! Here it is: " + reply("Bob owns 3 cats.") + " Hope that helps."
    assert parse_response(raw, _req()).rewritten_text == "Bob owns 3 cats."


@pytest.mark.parametrize("raw, msg", [
    ("not json at all", "no JSON"),
    (json.dumps({"applied": []}), "rewritten_text"),
    (json.dumps({"rewritten_text": "Bob owns 3 cats.", "applied": "e1"}), "applied"),
    (reply("Robert owns 3 cats."), "protected"),
    (reply("Bob owns 3 cats. " * 10), "exceeds"),
])
def test_parse_rejects_bad_output(raw, msg):
    with pytest.raises(OutputFormatError, match=msg):
        parse_response(raw, _req())


def test_unknown_ids_become_warnings_not_errors():
    out = parse_response(reply("Bob owns 3 cats.", applied=["zz"]), _req())
    assert out.warnings and "zz" in out.warnings[0]


# ── pipeline ─────────────────────────────────────────────────────
def test_structured_success_records_model_claims():
    backend = ScriptedBackend(reply("We met to talk.", applied=[], uncertain=["tone"]))
    result = RewriterPipeline(CFG, backend).rewrite_text(RewriteRequest(text="We met in order to talk."))
    assert result.status == RewriteStatus.SUCCESS
    assert result.rewritten_text == "We met to talk."
    cr = result.chunk_results[0]
    assert cr.uncertain == ["tone"] and cr.attempts == 1
    call = backend.calls[0]
    assert call["json_schema"] == OUTPUT_SCHEMA
    assert json.loads(call["prompt"])["text"] == "We met in order to talk."


def test_invalid_output_is_repaired_on_retry():
    backend = ScriptedBackend("garbage", reply("We met to talk."))
    result = RewriterPipeline(CFG, backend).rewrite_text(RewriteRequest(text="We met in order to talk."))
    assert result.rewritten_text == "We met to talk."
    assert result.chunk_results[0].attempts == 2
    assert "rejected" in backend.calls[1]["prompt"]
    assert backend.calls[1]["history"][1]["content"] == "garbage"


def test_persistent_bad_output_falls_back_to_rule_text():
    cfg = RewriterConfig(format_retries=1, intensity=Intensity.MODERATE)
    backend = ScriptedBackend("bad", "still bad")
    text = "We met in order to discuss the end result."
    result = RewriterPipeline(cfg, backend).rewrite_text(RewriteRequest(text=text, seed=1))
    assert result.status == RewriteStatus.FAILED
    assert result.rewritten_text == result.rule_text == "We met to discuss the result."
    assert "invalid output" in result.chunk_results[0].error


def test_rule_suggestions_are_sent_to_model():
    backend = ScriptedBackend(reply("We met to discuss the result."))
    RewriterPipeline(RewriterConfig(), backend).rewrite_text(
        RewriteRequest(text="We met in order to discuss the end result.", seed=1))
    sent = json.loads(backend.calls[0]["prompt"])
    assert sent["text"] == "We met in order to discuss the end result."  # original, not pre-edited
    assert {s["from"] for s in sent["suggestions"]} >= {"in order to", "end result"}


def test_memory_error_stops_further_calls():
    cfg = RewriterConfig(rules_enabled=False, max_prompt_tokens=5)
    backend = ScriptedBackend(LLMMemoryError("oom"))
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one."
    result = RewriterPipeline(cfg, backend).rewrite_text(RewriteRequest(text=text))
    assert len(backend.calls) == 1
    assert result.status == RewriteStatus.FAILED
    assert all(cr.rewritten_text == cr.chunk.text for cr in result.chunk_results)


# ── Ollama payload / errors (fake HTTP server) ───────────────────
def _backend(handler, **cfg):
    b = OllamaBackend(RewriterConfig(max_retries=0, **cfg))
    b._client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return b


def test_ollama_payload_enforces_memory_and_format():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    _backend(handler, context_window=2048, num_threads=4).generate(
        "hi", system="s", json_schema=OUTPUT_SCHEMA, history=[{"role": "user", "content": "prev"}])
    assert seen["options"]["num_ctx"] == 2048
    assert seen["options"]["num_thread"] == 4
    assert seen["format"] == OUTPUT_SCHEMA
    assert seen["think"] is False and seen["keep_alive"] == "5m"
    assert [m["content"] for m in seen["messages"]] == ["s", "prev", "hi"]


def test_ollama_out_of_memory_raises_memory_error():
    b = _backend(lambda r: httpx.Response(500, text='{"error":"model requires more system memory (5.1 GiB)"}'))
    with pytest.raises(LLMMemoryError):
        b.generate("hi")


def test_ollama_missing_model_gives_pull_hint():
    b = _backend(lambda r: httpx.Response(404, text='{"error":"model not found"}'))
    with pytest.raises(LLMError, match="ollama pull"):
        b.generate("hi")
