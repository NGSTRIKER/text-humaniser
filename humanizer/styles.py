"""Writing styles (modes): neutral, casual, professional, academic.

A style changes three things:

* **Word choice** in the synonym rule — which registers are allowed and
  which get a small preference.  Academic/professional never pick ornate
  words ("utilize", "aforementioned"): the goal is *measured* formality,
  not long words for their own sake.
* **Contractions** — professional/academic expand safe ones
  ("don't" → "do not").
* **Model instruction** — the tone line sent to the local model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Style = Literal["neutral", "casual", "professional", "academic"]
STYLES: tuple[str, ...] = ("neutral", "casual", "professional", "academic")


@dataclass(frozen=True)
class StyleProfile:
    name: str
    allow_informal: bool
    allow_ornate: bool
    elevated_bonus: float      # added to a synonym's ranking if its register is "elevated"
    expand_contractions: bool
    instruction: str


PROFILES: dict[str, StyleProfile] = {
    "neutral": StyleProfile(
        "neutral", False, False, 0.0, False,
        "Keep the original tone. Use clear, natural, everyday wording.",
    ),
    "casual": StyleProfile(
        "casual", True, False, -0.1, False,
        "Friendly, relaxed tone. Contractions are fine. Plain words; no slang that changes meaning.",
    ),
    "professional": StyleProfile(
        "professional", False, False, 0.05, True,
        "Professional business tone: clear, direct and polite. No slang, no contractions, "
        "no jargon or inflated words. Prefer short, concrete wording.",
    ),
    "academic": StyleProfile(
        "academic", False, False, 0.1, True,
        "Academic tone with measured formality: precise, objective and well connected. "
        "No slang or contractions. Use a slightly more formal word only when it is more "
        "precise; do NOT replace ordinary words with complex ones for their own sake "
        "(write 'use', not 'utilize'; 'team', not 'squad'). Keep sentences readable and "
        "keep hedges such as 'may' or 'suggests' exactly as strong as in the original.",
    ),
}


def get_profile(style: str | None) -> StyleProfile:
    try:
        return PROFILES[style or "neutral"]
    except KeyError:
        raise ValueError(f"unknown style {style!r}; choose from {STYLES}") from None


def effective_tone(style: str | None, tone: str | None) -> str:
    """Model instruction: the style's line, plus any custom tone the user gave."""
    base = get_profile(style).instruction
    if tone and tone.strip() and tone.strip().lower() != "keep the original tone":
        return f"{base} Additional instruction: {tone.strip()}"
    return base
