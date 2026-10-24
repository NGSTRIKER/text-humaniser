"""Structured LLM requests (Stage 7, plan §11).

Builds a compact JSON request containing the chunk text, a trimmed
linguistic analysis, rule suggestions with confidence, protected facts,
constraints and the required output schema.  The request is shrunk to
fit the token budget (analysis detail first, then low-confidence
suggestions) so the prompt + system prompt + output never exceed the context window.
"""

from __future__ import annotations

import json
from collections import Counter
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
    Intensity.MODERATE: "Improve readability, cadence, and flow. Vary sentence structure and openings where beneficial. Apply contextual improvements.",
    Intensity.HEAVY: "Substantially improve natural flow, variety, and structure. Restructure sentences, combine or split clauses, vary openings, and reorder where helpful while preserving all facts.",
}

SYSTEM_PROMPT = (
    "You are a skilled text editor. Reply with ONE compact JSON object matching `output_schema`, "
    "nothing else (no markdown, no <think> tags). Output JSON directly without thinking out loud.\n"
    "Rules: preserve every fact, name, number, date, negation and relationship. Never invent claims. "
    "Keep `protected` exact. Apply `suggestions` and `structural_guidance` to improve fluency, "
    "sentence variety and cadence. List accepted suggestion ids in `applied`. "
    "Keep `rejected` and `uncertain` empty ([]) or minimal to respond quickly. Stay within `max_output_chars`."
)


@dataclass
class LLMRewriteRequest:
    """Everything the LLM is told about one chunk."""

    text: str
    intensity: Intensity
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    analysis: dict[str, Any] = field(default_factory=dict)
    structural_guidance: list[str] = field(default_factory=list)
    tone: str = "keep the original tone"
    max_output_chars: int = 0

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
                "do not add new claims or invent facts",
                "actively vary structure, openings, and vocabulary across 70-80% of sentences",
                "reorder independent sentences within paragraphs or rearrange paragraphs where appropriate",
            ],
            "max_output_chars": self.max_output_chars,
            "output_schema": OUTPUT_SCHEMA,
        }
        if self.structural_guidance:
            payload["structural_guidance"] = self.structural_guidance
        return payload

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
                "to": e.replacement, "confidence": round(e.confidence, 2),
            })
    out.sort(key=lambda s: -s["confidence"])
    return out[:8]


def summarize_analysis(analysis: Any, chunk_start: int, chunk_end: int, limit: int = 6) -> dict[str, Any]:
    """Compact, chunk-local view of a :class:`DocumentAnalysis` (or ``{}``)."""
    if analysis is None:
        return {}

    def inside(x: Any) -> bool:
        return chunk_start <= x.char_start and x.char_end <= chunk_end

    sents = [s for s in analysis.sentences if inside(s)]
    chunk_openings: list[str] = []
    if sents:
        openings_list = [s.opening.lower() for s in sents if getattr(s, "opening", None)]
        counts = Counter(openings_list)
        chunk_openings = [o for o, c in counts.items() if c > 1][:limit]

    res: dict[str, Any] = {}
    passives = [s.text for s in sents if getattr(s, "voice", "") == "passive"][:limit]
    if passives:
        res["passive_sentences"] = passives
    entities = sorted({e.text for e in analysis.entities if inside(e)})[:limit]
    if entities:
        res["entities"] = entities
    numbers = sorted({q.text for q in analysis.quantities if inside(q)})[:limit]
    if numbers:
        res["numbers"] = numbers
    negations = sorted({f"{n.cue} {n.negated}" for n in analysis.negations if inside(n)})[:limit]
    if negations:
        res["negations"] = negations
    if chunk_openings:
        res["repeated_openings"] = chunk_openings
    return res


def derive_structural_guidance(
    edits: list[EditRecord],
    analysis: Any,
    chunk_start: int,
    chunk_end: int,
    intensity: Intensity,
    sentence_reordering: bool = False,
) -> list[str]:
    """Generate concise structural guidance for the LLM based on rule findings and linguistic metrics."""
    guidance: list[str] = []

    # 1. Structural rule candidates proposed for this chunk (whether accepted or skipped for overlap)
    inside_edits = [
        e for e in edits
        if chunk_start <= e.char_start and e.char_end <= chunk_end
    ]

    restructurings = [
        e for e in inside_edits
        if e.transform_type in {"word_order", "voice", "sentence_structure", "paragraph_flow"}
    ]
    for r in restructurings[:4]:
        if r.transform_type == "word_order":
            guidance.append(f"Phrase order: consider rearranging '{r.original_span}' → '{r.replacement}' ({r.reason}).")
        elif r.transform_type == "voice":
            guidance.append(f"Voice: consider active/passive conversion for '{r.original_span}' → '{r.replacement}'.")
        elif r.transform_type == "sentence_structure":
            if "combine" in r.rule_id:
                guidance.append("Sentence flow: combine short adjacent clauses with the same subject to avoid a choppy rhythm.")
            elif "split" in r.rule_id:
                guidance.append("Sentence flow: split long multi-clause sentence into clearer, well-paced sentences.")
        elif r.transform_type == "paragraph_flow":
            if "paragraph_swap" in r.rule_id:
                guidance.append("Document structure: you may swap independent paragraphs to improve logical progression.")
            else:
                guidance.append("Sentence flow: you may swap adjacent independent sentences to improve logical progression.")

    # 2. Linguistic analysis insights (openings, rhythm, repetition)
    if analysis is not None:
        sents = [s for s in analysis.sentences if chunk_start <= s.char_start and s.char_end <= chunk_end]
        if len(sents) >= 2:
            openings = [s.opening.lower() for s in sents if getattr(s, "opening", None)]
            repeated = [o for o, c in Counter(openings).items() if c > 1]
            if repeated:
                guidance.append(f"Sentence openings: vary beginnings; multiple sentences start with '{repeated[0]}'.")

            lengths = [len(s.text.split()) for s in sents]
            if len(lengths) >= 3 and all(l < 10 for l in lengths):
                guidance.append("Rhythm: several very short sentences in a row; combine related ideas for smoother cadence.")
            elif any(l > 30 for l in lengths):
                guidance.append("Rhythm: break up sentences exceeding 30 words into distinct statements.")

    # 3. Intensity-specific encouragement
    if intensity == Intensity.HEAVY:
        guidance.append("Restructuring goal: vary structure, openings, and cadence across 70-80% of sentences while strictly keeping facts.")
    elif intensity == Intensity.MODERATE:
        guidance.append("Rewrite moderately: improve phrasing and sentence flow while keeping the text natural and accurate.")

    if sentence_reordering:
        guidance.append("Paragraph structure: reorder independent sentences if doing so creates a more natural topic sequence.")

    # Deduplicate while preserving order
    return list(dict.fromkeys(guidance))[:6]


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
    structural_guidance: Optional[list[str]] = None,
) -> LLMRewriteRequest:
    """Create a request and trim it until it fits :func:`prompt_budget`."""
    summary = analysis_summary or {}
    req = LLMRewriteRequest(
        text=text,
        intensity=intensity or cfg.intensity,
        suggestions=list(suggestions or [])[: cfg.llm_max_suggestions],
        protected=protected_for_chunk(text, summary, list(protected_terms or [])),
        analysis=summary,
        structural_guidance=list(structural_guidance or []),
        tone=effective_tone(cfg.style, cfg.tone),
        max_output_chars=int(len(text) * cfg.max_expansion_ratio) + 20,
    )
    budget = prompt_budget(cfg)
    while cfg.estimate_tokens(req.to_prompt()) > budget:
        if req.structural_guidance:
            req.structural_guidance.pop()
        elif req.analysis:
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
