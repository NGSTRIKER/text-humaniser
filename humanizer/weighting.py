"""Candidate scoring, gating and probabilistic selection (Stage 6, plan §8).

score = type_multiplier · (w_u·usefulness + w_c·context_fit + w_g·grammar_fit
        + w_d·diversity) − w_m·meaning_risk − w_r·grammar_risk − w_t·domain_risk

Hard gates run BEFORE any randomness: a candidate that fails grammar,
touches a protected span, has no replacement, is disabled at the current
intensity, exceeds the profile's meaning-risk ceiling, or scores below the
threshold is rejected and can never be "rescued" by the random choice.

Selection then repeatedly draws one candidate from the acceptable pool
(softmax / linear / uniform, or greedy in deterministic mode), removes
everything that conflicts with it (see :mod:`humanizer.conflicts`) and
stops at the edit budget.  The probability each pick had is recorded.
"""

from __future__ import annotations

import random
import re
from typing import Optional, Sequence

from humanizer.candidates import Candidate, protection_conflict
from humanizer.config import Intensity, RewriterConfig
from humanizer.conflicts import conflict, conflict_message
from humanizer.probability import selection_probabilities, weighted_choice
from humanizer.profiles import PROFILES, STRUCTURAL_TYPES, IntensityProfile, resolve_profile

# Backwards-compatible alias (Stage 3 name).
LIMITS = PROFILES

__all__ = ["LIMITS", "STRUCTURAL_TYPES", "edit_cap", "gate", "score", "select", "structural_cap"]


def structural_cap(sentence_count: int, intensity: Intensity) -> int:
    """Max whole-sentence restructurings, so not every sentence is rewritten."""
    return PROFILES[intensity].structural_cap(sentence_count)


def edit_cap(word_count: int, intensity: Intensity) -> int:
    return PROFILES[intensity].edit_cap(word_count)


def score(c: Candidate, cfg: RewriterConfig, profile: Optional[IntensityProfile] = None) -> float:
    """Transparent weighted score; also stored on the candidate."""
    mult = profile.multiplier(c.transform_type) if profile else 1.0
    benefit = (
        cfg.weight_usefulness * c.usefulness
        + cfg.weight_context_fit * c.context_fit
        + cfg.weight_grammar_fit * c.confidence
        + cfg.weight_diversity * c.diversity_gain
    )
    risk = (
        cfg.weight_meaning_risk * c.meaning_risk
        + cfg.weight_grammar_risk * c.grammar_risk
        + cfg.weight_domain_risk * c.domain_risk
    )
    c.score = round(mult * benefit - risk, 4)
    return c.score


def gate(
    c: Candidate,
    cfg: RewriterConfig,
    intensity: Intensity,
    protected: list[tuple[int, int, str]],
    profile: Optional[IntensityProfile] = None,
) -> str | None:
    """Return a rejection reason, or ``None`` if the candidate is acceptable."""
    profile = profile or resolve_profile(intensity, cfg)
    if c.replacement is None:
        return f"flag only: {c.note or 'no safe replacement'}"
    if c.replacement == c.original:
        return "replacement identical to original"
    if not c.grammar_ok:
        return f"failed grammar check: {c.note}"
    conflict_reason = protection_conflict(c, protected)
    if conflict_reason:
        return conflict_reason
    if profile.multiplier(c.transform_type) <= 0:
        return f"{c.transform_type} edits disabled at {intensity.value} intensity"
    if c.meaning_risk > profile.max_meaning_risk:
        return (f"meaning risk {c.meaning_risk:.2f} exceeds {intensity.value} "
                f"limit {profile.max_meaning_risk:.2f}")
    threshold = cfg.min_edit_score + profile.min_score_offset
    if score(c, cfg, profile) < threshold:
        return f"score {c.score:.2f} below minimum {threshold:.2f}"
    return None


def _subsumed(candidates: list[Candidate]) -> tuple[list[Candidate], list[tuple[Candidate, str]]]:
    """Drop a shorter edit fully inside a longer edit."""
    keep, rejected = [], []
    for c in candidates:
        bigger = next(
            (d for d in candidates
             if d is not c
             and d.char_start <= c.char_start and c.char_end <= d.char_end
             and d.length > c.length
             and (d.transform_type == c.transform_type
                  or (d.transform_type in {"wordiness", "phrase", "redundancy"} and c.transform_type == "synonym"))
             and d.score >= c.score - 0.25),
            None,
        )
        if bigger:
            rejected.append((c, f"subsumed by larger edit {bigger.rule_id}"))
        else:
            keep.append(c)
    return keep, rejected


def _apply_word_to_replacement(text: str, orig: str, repl: str) -> str:
    """Case-preserving replacement of a word inside replacement text."""
    if not orig or not repl:
        return text
    pattern = re.compile(r"\b" + re.escape(orig) + r"\b", re.IGNORECASE)

    def _sub(m: re.Match) -> str:
        w = m.group(0)
        if w.isupper() and len(w) > 1:
            return repl.upper()
        if w[0].isupper():
            return repl.capitalize()
        return repl

    return pattern.sub(_sub, text, count=1)


def select(
    candidates: list[Candidate],
    cfg: RewriterConfig,
    cap: int,
    rng: random.Random,
    structural_limit: int | None = None,
    sentence_bounds: Sequence[tuple[int, int]] = (),
    profile: Optional[IntensityProfile] = None,
) -> tuple[list[Candidate], list[tuple[Candidate, str]]]:
    """Pick mutually compatible candidates, at most ``cap`` of them.

    Deterministic mode takes the best score each round (ties → priority,
    longer span, earlier position).  Otherwise draws using
    ``cfg.selection_method`` and the profile's temperature (falling back
    to ``cfg.selection_temperature``) with ``rng`` — seeded → reproducible.

    Returns ``(selected, [(rejected, reason), …])``; each selected
    candidate's ``selection_probability`` holds the chance it had.
    """
    temperature = profile.temperature if profile else cfg.selection_temperature
    remaining, rejected = _subsumed(candidates)
    remaining.sort(key=lambda c: (c.char_start, -c.length, c.rule_id))
    selected: list[Candidate] = []
    structural_used = 0

    while remaining and len(selected) < cap:
        if structural_limit is not None and structural_used >= structural_limit:
            for c in remaining:
                if c.transform_type in STRUCTURAL_TYPES:
                    rejected.append((c, f"restructuring limit reached ({structural_limit})"))
            remaining = [c for c in remaining if c.transform_type not in STRUCTURAL_TYPES]
            if not remaining:
                break

        pool = (
            [c for c in remaining if c.transform_type in STRUCTURAL_TYPES]
            if (structural_limit is not None and structural_used < structural_limit)
            else remaining
        )
        if not pool:
            pool = remaining

        if cfg.deterministic:
            pick = max(pool, key=lambda c: (c.score, c.priority, c.length, -c.char_start))
            pick.selection_probability = 1.0
        else:
            probs = selection_probabilities([c.score for c in pool], cfg.selection_method, temperature)
            pick = weighted_choice(pool, probs, rng)
            pick.selection_probability = round(probs[pool.index(pick)], 4)
        selected.append(pick)
        if pick.transform_type in STRUCTURAL_TYPES:
            structural_used += 1

        # If pick is structural, apply eligible word-level replacements inside pick.replacement
        if pick.transform_type in STRUCTURAL_TYPES and pick.replacement:
            for c in list(remaining):
                if (
                    c is not pick
                    and pick.char_start <= c.char_start and c.char_end <= pick.char_end
                    and c.replacement and c.original
                    and c.transform_type in {"synonym", "modifier", "wordiness"}
                ):
                    updated = _apply_word_to_replacement(pick.replacement, c.original, c.replacement)
                    if updated != pick.replacement:
                        pick.replacement = updated
                        pick.reason += f" (also updated '{c.original}' → '{c.replacement}')"

        keep = []
        for c in remaining:
            if c is pick:
                continue
            kind = conflict(c, pick, sentence_bounds)
            if kind:
                rejected.append((c, conflict_message(kind, pick.rule_id)))
            else:
                keep.append(c)
        remaining = keep

    for c in remaining:
        rejected.append((c, f"edit limit reached ({cap})"))
    selected.sort(key=lambda c: c.char_start)
    return selected, rejected
