"""Configuration management for the Rewriter AI system.

Uses pydantic-settings so values can come from environment variables,
a .env file, or be overridden programmatically in tests.
"""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Intensity(str, Enum):
    """Rewrite intensity level."""

    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"


class RewriterConfig(BaseSettings):
    """Central configuration for the entire rewriter pipeline.

    All fields have sensible defaults for 8 GB RAM / Intel iGPU.
    Override via env vars prefixed with ``REWRITER_`` or pass directly.
    """

    model_config = {"env_prefix": "REWRITER_", "case_sensitive": False}

    # ── LLM backend ──────────────────────────────────────────────
    model_name: str = Field(
        default="qwen3.5:4b",
        description="Ollama model tag to use for rewriting.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama HTTP API.",
    )
    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="Generation temperature. Lower = more deterministic.",
    )
    max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Maximum tokens the LLM may generate per request.",
    )
    context_window: int = Field(
        default=2048,
        gt=0,
        description="Advertised context window of the chosen model (tokens).",
    )
    request_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Seconds to wait for an Ollama response before timing out.",
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        description="How many times to retry a failed LLM call.",
    )

    # ── Chunking ─────────────────────────────────────────────────
    max_prompt_tokens: int = Field(
        default=160,
        gt=0,
        description=(
            "Max estimated tokens of *input text* per LLM request. "
            "System-prompt overhead is added on top."
        ),
    )
    system_prompt_token_budget: int = Field(
        default=400,
        ge=0,
        description="Estimated tokens consumed by the system prompt.",
    )
    chars_per_token: float = Field(
        default=4.0,
        gt=0,
        description="Rough character-to-token ratio for estimation.",
    )

    # ── Rewriting behaviour ──────────────────────────────────────
    intensity: Intensity = Field(
        default=Intensity.MODERATE,
        description="Rewrite intensity: light, moderate, or heavy.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. None = non-deterministic.",
    )

    # ── Logging ──────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Optional path to a log file. None = console only.",
    )

    # ── Validation thresholds (stubs for later stages) ───────────
    min_semantic_similarity: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum semantic similarity for a rewrite to be accepted.",
    )
    validation_enabled: bool = Field(default=True, description="Validate rewrites before accepting them.")
    semantic_strictness: Literal["low", "medium", "high"] = Field(
        default="medium", description="How many non-critical problems block a rewrite (3 / 2 / 1)."
    )
    max_retry_count: int = Field(
        default=2,
        ge=0,
        description="Max validation→repair→revalidate cycles.",
    )

    # ── Rule engine (Stage 3) ────────────────────────────────────
    rules_enabled: bool = Field(default=True, description="Run the rule engine before the LLM.")
    rules_only: bool = Field(default=False, description="Skip the LLM; output rule-engine edits only.")
    deterministic: bool = Field(
        default=False,
        description="Always pick the best-scoring edit instead of weighted random choice.",
    )
    selection_temperature: float = Field(
        default=0.5, gt=0.0, description="Softmax temperature for weighted edit selection."
    )
    min_edit_score: float = Field(default=0.4, description="Candidates scoring below this are rejected.")
    weight_usefulness: float = Field(default=1.0, ge=0.0)
    weight_context_fit: float = Field(default=0.5, ge=0.0)
    weight_grammar_fit: float = Field(default=0.5, ge=0.0)
    weight_diversity: float = Field(default=0.3, ge=0.0)
    weight_meaning_risk: float = Field(default=1.5, ge=0.0)
    weight_grammar_risk: float = Field(default=1.0, ge=0.0)

    # ── Local LLM integration (Stage 7) ──────────────────────────
    llm_mode: Literal["structured", "plain"] = Field(
        default="structured",
        description="structured = JSON request/response with suggestions; plain = text in, text out.",
    )
    tone: str = Field(default="keep the original tone", description="Extra tone/style instruction for the LLM.")
    style: Literal["neutral", "casual", "professional", "academic"] = Field(
        default="neutral",
        description="Writing mode: affects word choice, contractions and the model's instruction.",
    )
    max_expansion_ratio: float = Field(
        default=1.3, gt=0.5, le=3.0, description="Rewrite may be at most this many times the input length."
    )
    llm_max_suggestions: int = Field(default=20, ge=0, description="Max rule suggestions sent per chunk.")
    format_retries: int = Field(default=1, ge=0, description="Re-asks after invalid structured output.")
    keep_alive: str = Field(default="5m", description="How long Ollama keeps the model loaded.")
    unload_model_on_close: bool = Field(default=False, description="Free model memory when the run ends.")
    llm_think: bool = Field(default=False, description="Allow 'thinking' output on models that support it.")
    num_threads: Optional[int] = Field(default=None, ge=1, description="CPU threads for Ollama (None = auto).")

    # ── Probabilistic selection (Stage 6) ────────────────────────
    weight_domain_risk: float = Field(default=1.0, ge=0.0, description="Penalty for touching technical terms.")
    selection_method: Literal["softmax", "linear", "uniform"] = Field(
        default="softmax", description="How scores become selection probabilities."
    )
    randomness: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="0 = near-greedy, 1 = very varied. None = intensity profile default.",
    )
    max_edits: Optional[int] = Field(default=None, ge=1, description="Absolute cap on applied edits.")
    max_edit_ratio: Optional[float] = Field(
        default=None, gt=0.0, le=1.0, description="Cap on edits per word (0.1 = 10 per 100 words)."
    )
    vocabulary_variation: Literal["low", "medium", "high"] = Field(default="medium")
    restructuring: Literal["low", "medium", "high"] = Field(default="medium")
    type_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Per transform-type score multiplier, e.g. {\"voice\": 0.5}; 0 disables a type.",
    )

    # ── Word graph / synonyms (Stage 4) ──────────────────────────
    synonyms_enabled: bool = Field(default=True, description="Vary repeated words using WordNet.")
    min_sense_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Skip a word when its meaning in context is less certain than this.",
    )
    use_embeddings: bool = Field(
        default=False,
        description="Rank synonyms with spaCy word vectors (needs en_core_web_md or _lg).",
    )
    embedding_model: str = Field(default="en_core_web_md")

    # ── Structural transformations (Stage 5) ─────────────────────
    voice_mode: Literal["disabled", "selective", "frequent"] = Field(
        default="selective",
        description="Active/passive conversion: disabled, selective, or frequent.",
    )
    sentence_reordering: bool = Field(
        default=False,
        description="Allow swapping logically independent sentences (heavy only). Off by default.",
    )


    @field_validator("type_weights")
    @classmethod
    def _check_type_weights(cls, v: dict[str, float]) -> dict[str, float]:
        bad = {k: w for k, w in v.items() if w < 0}
        if bad:
            raise ValueError(f"type_weights must be >= 0, got {bad}")
        return v

    @classmethod
    def from_weights_file(cls, path: str | Path, **overrides: Any) -> "RewriterConfig":
        """Load selection weights/settings from a JSON file, then apply overrides.

        The file holds any subset of config fields, e.g.
        ``{"weight_meaning_risk": 2.0, "type_weights": {"voice": 0.5}}``.
        Unknown keys raise ``ValueError`` so typos are not silently ignored.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read weights file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("weights file must contain a JSON object")
        unknown = set(data) - set(cls.model_fields)
        if unknown:
            raise ValueError(f"unknown settings in weights file: {sorted(unknown)}")
        return cls(**{**data, **overrides})

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, v: str) -> str:
        v = v.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return v

    # ── Derived helpers ──────────────────────────────────────────

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate from character count."""
        return max(1, int(len(text) / self.chars_per_token))

    def available_input_tokens(self) -> int:
        """How many input-text tokens fit in the context window after
        reserving space for the system prompt and the generation budget."""
        return max(
            256,
            self.context_window
            - self.system_prompt_token_budget
            - self.max_tokens,
        )
