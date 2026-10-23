"""Structured LLM requests (Stage 7, plan §11).

Builds a compact JSON request containing the chunk text, a trimmed
linguistic analysis, rule suggestions with confidence, protected facts,
constraints and the required output schema.  The request is shrunk to
fit the token budget (analysis detail first, then low-confidence
suggestions) so the prompt + system prompt + output never exceed the context window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from humanizer.styles import effective_tone
from humanizer.config import Intensity, RewriterConfig
from humanizer.schemas import EditRecord

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rewritten_text": {"type": "string"},
        "applied": {"type": "array", "items": {"type": "string"}},
        "rejected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["id", "reason"],
            },
        },
        "uncertain": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["rewritten_text", "applied", "rejected", "uncertain"],
}

_INTENSITY_GOALS = {
    Intensity.LIGHT: "Fix obvious redundancy, wordiness and grammar. Keep most wording and structure.",
    Intensity.MODERATE: "Improve readability and flow. Contextual word changes and selective restructuring are allowed.",
    Intensity.HEAVY: "Broader restructuring, combining or splitting sentences and more stylistic variety are allowed.",
}

SYSTEM_PROMPT = (
    "You are a careful local text editor. You receive a JSON request and reply with ONE JSON "
    "object matching `output_schema`, nothing else (no markdown, no commentary).\n"
    "Rules: keep every fact, name, number, date, negation, uncertainty word and cause/effect "
    "relationship. Never invent information. Keep every string in `protected` exactly. "
    "`suggestions` are optional rule-based edits: apply the good ones, reject unsuitable ones "
    "with a short reason, and list their ids in `applied` or `rejected`. Put anything you are "
    "unsure about in `uncertain`. Stay within `max_output_chars`."
)


@dataclass
class LLMRewriteRequest:
    """Everything the LLM is told about one chunk."""

    text: str
    intensity: Intensity
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    analysis: dict[str, Any] = field(default_factory=dict)
    tone: str = "keep the original tone"
    max_output_chars: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "task": "rewrite",
            "intensity": self.intensity.value,
            "goal": _INTENSITY_GOALS[self.intensity],
            "tone": self.tone,
            "text": self.text,
            "protected": self.protected,
            "suggestions": self.suggestions,
            "analysis": self.analysis,
            "constraints": [
                "preserve meaning, facts, negation and uncertainty",
                "do not add new claims, examples or connectors like 'because' unless already implied",
                "do not drop details",
            ],
            "max_output_chars": self.max_output_chars,
            "output_schema": OUTPUT_SCHEMA,
        }

    def to_prompt(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False, separators=(",", ":"))


def suggestions_for_chunk(
    edits: list[EditRecord], chunk_start: int, chunk_end: int
) -> list[dict[str, Any]]:
    """Accepted rule edits lying fully inside the chunk, highest confidence first."""
    out = []
    for e in edits:
        if not (e.accepted and e.user_decision is not False):
            continue
        if chunk_start <= e.char_start and e.char_end <= chunk_end:
            out.append({
                "id": e.edit_id, "type": e.transform_type, "from": e.original_span,
                "to": e.replacement, "why": e.reason, "confidence": round(e.confidence, 2),
            })
    out.sort(key=lambda s: -s["confidence"])
    return out


def summarize_analysis(analysis: Any, chunk_start: int, chunk_end: int, limit: int = 8) -> dict[str, Any]:
    """Compact, chunk-local view of a :class:`DocumentAnalysis` (or ``{}``)."""
    if analysis is None:
        return {}

    def inside(x: Any) -> bool:
        return chunk_start <= x.char_start and x.char_end <= chunk_end

    sents = [s for s in analysis.sentences if inside(s)]
    return {
        "passive_sentences": [s.text for s in sents if s.voice == "passive"][:limit],
        "entities": sorted({e.text for e in analysis.entities if inside(e)})[:limit],
        "numbers": sorted({q.text for q in analysis.quantities if inside(q)})[:limit],
        "negations": sorted({f"{n.cue} {n.negated}" for n in analysis.negations if inside(n)})[:limit],
        "repeated_words": [w for w, _ in sorted(analysis.repeated_words.items(), key=lambda kv: -kv[1])][:limit],
        "sentence_count": len(sents),
    }


def protected_for_chunk(text: str, analysis_summary: dict[str, Any], terms: list[str]) -> list[str]:
    seen: list[str] = []
    for t in [*terms, *analysis_summary.get("entities", []), *analysis_summary.get("numbers", [])]:
        if t and t in text and t not in seen:
            seen.append(t)
    return seen


def prompt_budget(cfg: RewriterConfig) -> int:
    """Tokens the user message may use: context window minus system prompt and output."""
    return cfg.context_window - cfg.max_tokens - cfg.estimate_tokens(SYSTEM_PROMPT)


def build_request(
    text: str,
    cfg: RewriterConfig,
    *,
    intensity: Optional[Intensity] = None,
    suggestions: Optional[list[dict[str, Any]]] = None,
    analysis_summary: Optional[dict[str, Any]] = None,
    protected_terms: Optional[list[str]] = None,
) -> LLMRewriteRequest:
    """Create a request and trim it until it fits :func:`prompt_budget`."""
    summary = analysis_summary or {}
    req = LLMRewriteRequest(
        text=text,
        intensity=intensity or cfg.intensity,
        suggestions=list(suggestions or [])[: cfg.llm_max_suggestions],
        protected=protected_for_chunk(text, summary, list(protected_terms or [])),
        analysis=summary,
        tone=effective_tone(cfg.style, cfg.tone),
        max_output_chars=int(len(text) * cfg.max_expansion_ratio) + 20,
    )
    budget = prompt_budget(cfg)
    while cfg.estimate_tokens(req.to_prompt()) > budget:
        if req.analysis:
            req.analysis = {}
        elif req.suggestions:
            req.suggestions.pop()  # lowest confidence last
        else:
            break
    if cfg.estimate_tokens(req.to_prompt()) > budget:
        raise ValueError(
            "chunk does not fit the context window; lower max_prompt_tokens or raise context_window"
        )
    return req
