"""Stage 8: accept → repair → revalidate → fallback, inside the pipeline."""

import json

import pytest

from humanizer.analyzer import is_available
from humanizer.config import RewriterConfig
from humanizer.decision import decide
from humanizer.rewriter import RewriterPipeline
from humanizer.schemas import RewriteRequest, ValidationReport

pytestmark = pytest.mark.skipif(not is_available(), reason="spaCy model en_core_web_sm not installed")


def reply(text):
    return json.dumps({"rewritten_text": text, "applied": [], "rejected": [], "uncertain": []})


class ScriptedBackend:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def health_check(self):
        return True

    def generate(self, prompt, system="", **kw):
        self.calls.append(prompt)
        return self.replies.pop(0)


class FakeValidator:
    """Valid only for texts in ``good``."""

    def __init__(self, good):
        self.good, self.calls = set(good), 0

    def validate(self, original, text):
        self.calls += 1
        ok = text in self.good
        return ValidationReport(is_valid=ok, decision="accept" if ok else "repair",
                                errors=[] if ok else ["bad"], repair_suggestions=[] if ok else ["fix it"])


def test_decide_accepts_valid_candidate():
    text, rep, source, n = decide("o", "good", FakeValidator({"good"}))
    assert (text, source, n) == ("good", "llm", 0)


def test_decide_repairs_then_accepts():
    text, _, source, n = decide("o", "bad", FakeValidator({"fixed"}), repair=lambda c, r: "fixed")
    assert (text, source, n) == ("fixed", "repaired", 1)


def test_decide_bounded_retries_then_fallback():
    calls = []
    def repair(c, r):
        calls.append(c)
        return c + "!"
    text, rep, source, n = decide("o", "bad", FakeValidator({"rules"}), repair=repair,
                                  max_repairs=2, fallbacks=[("rules", "rules")])
    assert len(calls) == 2 and n == 2
    assert (text, source) == ("rules", "rules") and "rejected" in rep.warnings[0]


def test_decide_falls_back_to_original_last():
    text, rep, source, _ = decide("orig", "bad", FakeValidator(set()), fallbacks=[("rules", "also bad")])
    assert (text, source) == ("orig", "original") and rep.is_valid


CFG = RewriterConfig(rules_enabled=False, max_retry_count=1)
TEXT = "The defendant was not found guilty."


def test_pipeline_repairs_meaning_change():
    backend = ScriptedBackend(reply("The defendant was found guilty."),
                              reply("The defendant was not judged guilty."))
    result = RewriterPipeline(CFG, backend).rewrite_text(RewriteRequest(text=TEXT))
    cr = result.chunk_results[0]
    assert result.rewritten_text == "The defendant was not judged guilty."
    assert cr.source == "repaired" and cr.validation.is_valid
    repair_request = json.loads(backend.calls[1])
    assert repair_request["task"] == "repair" and repair_request["problems"]


def test_pipeline_falls_back_when_repair_fails():
    backend = ScriptedBackend(reply("The defendant was found guilty."),
                              reply("The defendant was declared guilty."))
    result = RewriterPipeline(CFG, backend).rewrite_text(RewriteRequest(text=TEXT))
    assert result.rewritten_text == TEXT
    assert result.chunk_results[0].source == "original"
    assert len(backend.calls) == 2  # 1 rewrite + max_retry_count repairs, no loop


def test_pipeline_uses_rule_version_when_model_breaks_meaning():
    cfg = RewriterConfig(max_retry_count=0)
    text = "We did not meet in order to discuss the end result."
    backend = ScriptedBackend(reply("We met to discuss the result."))
    result = RewriterPipeline(cfg, backend).rewrite_text(RewriteRequest(text=text, seed=1))
    assert result.rewritten_text == result.rule_text
    assert result.chunk_results[0].source == "rules"


def test_validation_can_be_disabled():
    backend = ScriptedBackend(reply("The defendant was found guilty."))
    cfg = RewriterConfig(rules_enabled=False, validation_enabled=False)
    result = RewriterPipeline(cfg, backend).rewrite_text(RewriteRequest(text=TEXT))
    assert result.rewritten_text == "The defendant was found guilty."
