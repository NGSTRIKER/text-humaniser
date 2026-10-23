"""Decision controller (§12 G): accept → repair → revalidate → fall back.

    text, report, source, repairs = decide(original, candidate, validator,
                                           repair=fn, fallbacks=[("rules", rule_text)])

* ``candidate`` is validated against ``original``.
* While it fails and repairs remain, ``repair(candidate, report)`` asks the
  LLM for a targeted fix; the result is validated again.
* After ``max_repairs`` failed repairs the first fallback that validates is
  used; the original text is the last resort (always safe).

Bounded by construction: at most ``max_repairs`` repair calls per chunk.
"""

from __future__ import annotations

from typing import Callable, Optional

from humanizer.logging_config import get_logger
from humanizer.schemas import ValidationReport

log = get_logger("decision")

RepairFn = Callable[[str, ValidationReport], Optional[str]]


def decide(
    original: str,
    candidate: str,
    validator,
    *,
    repair: Optional[RepairFn] = None,
    max_repairs: int = 2,
    fallbacks: list[tuple[str, str]] = (),
) -> tuple[str, ValidationReport, str, int]:
    """Return ``(text, report, source, repairs_used)``; source is llm | repaired | <fallback name> | original."""
    report = validator.validate(original, candidate)
    repairs = 0
    source = "llm"
    while not report.is_valid and repair is not None and repairs < max_repairs and report.decision == "repair":
        repairs += 1
        fixed = repair(candidate, report)
        if not fixed or fixed.strip() == candidate.strip():
            log.info("Repair %d produced nothing new; stopping", repairs)
            break
        candidate, source = fixed, "repaired"
        report = validator.validate(original, candidate)
        log.info("Repair %d → %s", repairs, "valid" if report.is_valid else "still invalid")
    if report.is_valid:
        return candidate, report, source, repairs

    log.warning("Rewrite rejected: %s", "; ".join(report.errors or report.warnings[:3]))
    for name, text in fallbacks:
        if text.strip() == original.strip():
            continue
        fb = validator.validate(original, text)
        if fb.is_valid:
            fb.warnings.insert(0, f"model rewrite rejected ({'; '.join(report.errors[:3]) or 'too many problems'}); using {name} version")
            return text, fb, name, repairs
    final = ValidationReport(
        is_valid=True, decision="accept", confidence=1.0,
        warnings=[f"model rewrite rejected ({'; '.join(report.errors[:3]) or 'too many problems'}); original kept"],
    )
    return original, final, "original", repairs
