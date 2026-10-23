"""Combine all checks into a :class:`ValidationReport` (§12 G decision policy).

Policy:
* any CRITICAL issue                → reject (decision ``repair`` if a repair is possible)
* MAJOR issues ≥ strictness limit   → reject / repair
* otherwise                         → accept (MINOR issues are reported only)

Strictness limits for MAJOR issues: low = 3, medium = 2, high = 1.
"""

from __future__ import annotations

from typing import Any, Optional

from humanizer.analyzer import AnalyzerUnavailableError, analyze
from humanizer.config import RewriterConfig
from humanizer.logging_config import get_logger
from humanizer.schemas import ValidationReport
from humanizer.validation.base import CheckResult, Pair, Severity
from humanizer.validation.events import check_events
from humanizer.validation.facts import check_additions_omissions, check_facts
from humanizer.validation.grammar import check_grammar
from humanizer.validation.similarity import check_similarity

log = get_logger("validation")

MAJOR_LIMIT = {"low": 3, "medium": 2, "high": 1}


class Validator:
    def __init__(self, config: Optional[RewriterConfig] = None, nlp: Any = None, vector_nlp: Any = None) -> None:
        self.config = config or RewriterConfig()
        self.nlp = nlp
        self.vector_nlp = vector_nlp
        if vector_nlp is None and self.config.use_embeddings:
            try:
                from humanizer.analyzer import load_nlp
                self.vector_nlp = load_nlp(self.config.embedding_model)
            except Exception as exc:  # optional
                log.info("Embeddings unavailable (%s); using lexical similarity", exc)

    def validate(self, original: str, rewritten: str) -> ValidationReport:
        if original.strip() == rewritten.strip():
            return ValidationReport(decision="accept", confidence=1.0)
        try:
            pair = Pair(analyze(original, nlp=self.nlp), analyze(rewritten, nlp=self.nlp))
        except AnalyzerUnavailableError as exc:
            return ValidationReport(
                is_valid=False, overall_score=0.0, decision="reject", confidence=0.0,
                errors=[f"validation unavailable: {exc}"],
                warnings=["cannot verify meaning without the analyzer; rewrite not accepted"],
            )
        checks = [
            check_facts(pair),
            check_events(pair),
            check_additions_omissions(pair),
            check_similarity(pair, self.config.min_semantic_similarity, self.vector_nlp),
            check_grammar(pair),
        ]
        return build_report(checks, self.config.semantic_strictness)


def build_report(checks: list[CheckResult], strictness: str = "medium") -> ValidationReport:
    issues = [i for c in checks for i in c.issues]
    critical = [i for i in issues if i.severity == Severity.CRITICAL]
    major = [i for i in issues if i.severity == Severity.MAJOR]
    minor = [i for i in issues if i.severity == Severity.MINOR]

    meaning_cats = {"number", "entity", "negation", "certainty", "condition", "causal", "comparison", "role", "polarity", "event"}
    rejected = bool(critical) or len(major) >= MAJOR_LIMIT[strictness]
    repairs = list(dict.fromkeys(i.repair for i in critical + major if i.repair))

    score = max(0.0, 1.0 - 0.4 * len(critical) - 0.15 * len(major) - 0.03 * len(minor))
    # Confidence falls when checks were skipped or many soft signals disagree.
    skipped = sum(c.skipped for c in checks)
    confidence = round(max(0.1, 1.0 - 0.15 * skipped - 0.05 * len(minor)), 2)

    by = {c.name: c for c in checks}
    sim = by.get("similarity")
    report = ValidationReport(
        is_valid=not rejected,
        overall_score=round(score, 3),
        fact_preservation=by["additions_omissions"].score if "additions_omissions" in by else None,
        entity_check_passed=not any(i.category == "entity" for i in critical),
        negation_preserved=not any(i.category in {"negation", "polarity"} for i in critical),
        semantic_similarity=sim.score if sim else None,
        grammar_ok=not any(i.category == "grammar" for i in major),
        unsupported_additions=[i.rewritten or i.message for i in issues if i.category in {"addition", "causal"} and i.rewritten],
        missing_information=[i.original or i.message for i in issues if i.category in {"omission", "entity", "number"} and i.original],
        suspected_meaning_changes=[i.message for i in critical + major if i.category in meaning_cats],
        grammar_problems=[i.message for i in issues if i.category in {"grammar", "tense", "style"}],
        warnings=[f"[{i.severity.value}] {i.message}" for i in major + minor],
        errors=[i.message for i in critical],
        checks={c.name: {"passed": c.passed, "score": c.score, "skipped": c.skipped, "note": c.note,
                         "issues": [f"[{i.severity.value}] {i.message}" for i in c.issues]} for c in checks},
        repair_suggestions=repairs,
        confidence=confidence,
        decision="accept" if not rejected else ("repair" if repairs else "reject"),
    )
    return report


def validate(original: str, rewritten: str, config: Optional[RewriterConfig] = None, nlp: Any = None) -> ValidationReport:
    return Validator(config, nlp=nlp).validate(original, rewritten)
