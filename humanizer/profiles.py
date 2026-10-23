"""Intensity profiles (Stage 6, plan §9).

A profile bundles every knob that changes with rewrite intensity:
risk ceiling, edit budget, restructuring share, selection temperature,
score threshold and per-transform-type multipliers.  User settings in
:class:`~humanizer.config.RewriterConfig` are layered on top by
:func:`resolve_profile`, so the base profiles never need editing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from humanizer.config import Intensity, RewriterConfig

LEXICAL_TYPES = frozenset({"synonym", "repetition", "phrase"})
STRUCTURAL_TYPES = frozenset({"voice", "word_order", "sentence_structure", "paragraph_flow"})


@dataclass(frozen=True)
class IntensityProfile:
    """All intensity-dependent selection limits.

    Attributes:
        name: Intensity this profile describes.
        max_meaning_risk: Hard ceiling; riskier candidates are rejected.
        edits_per_100_words: Edit budget as a rate.
        structural_share: Share of sentences that may be restructured.
        temperature: Softmax temperature (higher = more varied choices).
        min_score_offset: Added to ``min_edit_score`` (light is stricter).
        type_multipliers: Score multiplier per transform type; 0 disables it.
        max_edits: Absolute cap (``None`` = rate only).
    """

    name: Intensity
    max_meaning_risk: float
    edits_per_100_words: float
    structural_share: float
    temperature: float
    min_score_offset: float = 0.0
    type_multipliers: dict[str, float] = field(default_factory=dict)
    max_edits: int | None = None

    def multiplier(self, transform_type: str) -> float:
        return self.type_multipliers.get(transform_type, 1.0)

    def edit_cap(self, word_count: int) -> int:
        cap = max(1, math.ceil(word_count * self.edits_per_100_words / 100))
        return min(cap, self.max_edits) if self.max_edits is not None else cap

    def structural_cap(self, sentence_count: int) -> int:
        if self.structural_share <= 0:
            return 0
        return max(1, math.floor(sentence_count * self.structural_share + 0.5))


PROFILES: dict[Intensity, IntensityProfile] = {
    # Light: obvious redundancy/wordiness only; keep wording and structure.
    Intensity.LIGHT: IntensityProfile(
        Intensity.LIGHT, 0.1, 8, 0.0, 0.3, min_score_offset=0.1,
        type_multipliers={"synonym": 0.6, "word_order": 0.0, "voice": 0.0,
                          "sentence_structure": 0.0, "paragraph_flow": 0.0},
    ),
    # Moderate: readability, contextual vocabulary, selective restructuring.
    Intensity.MODERATE: IntensityProfile(Intensity.MODERATE, 0.2, 15, 0.34, 0.5),
    # Heavy: broader restructuring and variation, never more meaning risk than 0.35.
    Intensity.HEAVY: IntensityProfile(
        Intensity.HEAVY, 0.35, 25, 0.6, 0.8, min_score_offset=-0.05,
        type_multipliers={"synonym": 1.1, "sentence_structure": 1.1},
    ),
}

_LEVEL_SCALE = {"low": 0.7, "medium": 1.0, "high": 1.3}


def resolve_profile(intensity: Intensity, cfg: RewriterConfig) -> IntensityProfile:
    """Base profile for ``intensity`` with the user's config applied."""
    p = PROFILES[intensity]
    mult = dict(p.type_multipliers)

    vocab = _LEVEL_SCALE[cfg.vocabulary_variation]
    for t in LEXICAL_TYPES:
        mult[t] = mult.get(t, 1.0) * vocab
    restructure = _LEVEL_SCALE[cfg.restructuring]
    for t in STRUCTURAL_TYPES:
        mult[t] = mult.get(t, 1.0) * restructure
    for t, w in cfg.type_weights.items():  # explicit user weights win
        mult[t] = w

    temperature = p.temperature if cfg.randomness is None else max(0.05, cfg.randomness * 1.5)
    rate = p.edits_per_100_words
    if cfg.max_edit_ratio is not None:
        rate = min(rate, cfg.max_edit_ratio * 100)
    share = min(1.0, p.structural_share * restructure) if p.structural_share else 0.0

    return replace(
        p,
        type_multipliers=mult,
        temperature=temperature,
        edits_per_100_words=rate,
        structural_share=share,
        max_edits=cfg.max_edits,
    )
