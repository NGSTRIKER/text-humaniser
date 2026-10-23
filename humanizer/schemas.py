"""Typed data structures used across the rewriter pipeline.

These are the *contracts* that flow between stages.  Keeping them in one
place makes it easy to evolve the pipeline without breaking consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Text chunking ────────────────────────────────────────────────


@dataclass(frozen=True)
class TextChunk:
    """A contiguous slice of input text (usually one or more paragraphs).

    Attributes:
        index: Sequential chunk number (0-based).
        text: The raw text content of this chunk.
        char_start: Character offset of the first character in the
                     original document.
        char_end:   Character offset *past* the last character.
        paragraph_indices: Which paragraph numbers (0-based) this chunk
                           comprises.
    """

    index: int
    text: str
    char_start: int
    char_end: int
    paragraph_indices: tuple[int, ...] = ()


# ── LLM interaction ─────────────────────────────────────────────


@dataclass
class ChunkResult:
    """Result of running one chunk through the LLM.

    Attributes:
        chunk: The source chunk that was processed.
        rewritten_text: The LLM's output (or original text on failure).
        success: Whether the LLM call succeeded.
        error: Error message if ``success`` is False.
        model: Which model produced the rewrite.
        duration_seconds: Wall-clock time for the LLM call.
    """

    chunk: TextChunk
    rewritten_text: str
    success: bool = True
    error: Optional[str] = None
    model: Optional[str] = None
    duration_seconds: float = 0.0
    # Stage 7 structured-output details (model claims, not verified facts).
    applied_suggestions: list[str] = field(default_factory=list)
    rejected_suggestions: list[dict] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0
    validation: Optional["ValidationReport"] = None
    source: str = "llm"   # llm | rules | original — which version was finally used


# ── Rewrite request / result ────────────────────────────────────


class RewriteStatus(str, Enum):
    """Overall status of a rewrite operation."""

    SUCCESS = "success"
    PARTIAL = "partial"  # some chunks failed; originals kept
    FAILED = "failed"
    SKIPPED = "skipped"  # e.g. input too short or identical


@dataclass
class RewriteRequest:
    """Everything needed to kick off a rewrite.

    Attributes:
        text: The full input text.
        intensity: Desired rewrite intensity.
        protected_terms: Terms that must not be changed.
        seed: Random seed for reproducibility.
    """

    text: str
    intensity: str = "moderate"
    protected_terms: list[str] = field(default_factory=list)
    seed: Optional[int] = None


@dataclass
class RewriteResult:
    """The final output of the pipeline for one document.

    Attributes:
        original_text: Verbatim input.
        rewritten_text: Final assembled output.
        status: Overall outcome.
        chunk_results: Per-chunk details.
        total_chunks: Number of chunks the input was split into.
        successful_chunks: How many chunks were rewritten successfully.
        duration_seconds: Wall-clock time for the entire operation.
    """

    original_text: str
    rewritten_text: str
    status: RewriteStatus = RewriteStatus.SUCCESS
    chunk_results: list[ChunkResult] = field(default_factory=list)
    total_chunks: int = 0
    successful_chunks: int = 0
    duration_seconds: float = 0.0
    edits: list["EditRecord"] = field(default_factory=list)
    rule_text: Optional[str] = None  # text after the rule engine, before the LLM


# ── Edit tracking ───────────────────────────────────────────────


@dataclass
class EditRecord:
    """One proposed edit and what happened to it.

    ``char_start``/``char_end`` point into the ORIGINAL text.
    ``accepted`` is the engine's decision; ``user_decision`` is the
    user's (``None`` = no opinion, ``False`` = undone).
    ``validation`` is filled by the validator in Stage 8.
    """

    edit_id: str = ""
    original_span: str = ""
    replacement: str = ""
    rule_id: str = ""
    transform_type: str = ""
    reason: str = ""
    confidence: float = 0.0
    semantic_risk: float = 0.0
    accepted: Optional[bool] = None
    char_start: int = -1
    char_end: int = -1
    score: float = 0.0
    rejection_reason: Optional[str] = None
    validation: Optional[str] = None
    user_decision: Optional[bool] = None
    selection_probability: Optional[float] = None


# ── Validation (stub — populated in Stage 8) ────────────────────


@dataclass
class ValidationReport:
    """Structured validation report comparing original vs rewrite.

    Filled by :mod:`humanizer.validation` (Stage 8).  ``decision`` is
    ``accept`` | ``repair`` | ``reject``; ``checks`` holds per-check details.
    """

    is_valid: bool = True
    overall_score: float = 1.0
    fact_preservation: Optional[float] = None
    entity_check_passed: Optional[bool] = None
    negation_preserved: Optional[bool] = None
    semantic_similarity: Optional[float] = None
    grammar_ok: Optional[bool] = None
    unsupported_additions: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    suspected_meaning_changes: list[str] = field(default_factory=list)
    grammar_problems: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    repair_suggestions: list[str] = field(default_factory=list)
    confidence: float = 1.0
    decision: str = "accept"
