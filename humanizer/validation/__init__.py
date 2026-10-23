"""Semantic and structural validation (Stage 8, plan §12).

    >>> from humanizer.validation import validate
    >>> validate("John threw rocks into the lake.",
    ...          "Rocks were thrown into the lake by John.").is_valid
    True

Checks are independent modules returning :class:`CheckResult` objects:

* ``facts``      — entities, numbers, dates, negation, modality, conditions,
  comparisons, causal claims, additions and omissions (§12 A, D, F)
* ``events``     — who did what to whom, voice-aware (§12 B)
* ``similarity`` — optional embedding / lexical similarity signal (§12 C)
* ``grammar``    — re-analysis of the rewrite for introduced problems (§12 E)

:func:`validate` combines them under the decision policy (§12 G).
"""

from humanizer.validation.base import CheckResult, Issue, Severity
from humanizer.validation.validator import Validator, validate

__all__ = ["CheckResult", "Issue", "Severity", "Validator", "validate"]
