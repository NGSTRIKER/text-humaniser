"""Stage 8: semantic validation (formerly placeholder tests)."""

import pytest

from humanizer.analyzer import is_available
from humanizer.schemas import ValidationReport

needs_spacy = pytest.mark.skipif(not is_available(), reason="spaCy model en_core_web_sm not installed")


class TestValidationReport:
    def test_default_is_valid(self):
        report = ValidationReport()
        assert report.is_valid is True and report.overall_score == 1.0 and report.decision == "accept"

    def test_can_flag_invalid(self):
        report = ValidationReport(is_valid=False, overall_score=0.3)
        assert report.is_valid is False and report.overall_score < 0.5

    def test_can_add_warnings_and_errors(self):
        report = ValidationReport(is_valid=False, warnings=["w"], errors=["Negation flipped", "Number changed"])
        assert len(report.warnings) == 1 and len(report.errors) == 2


@needs_spacy
class TestSemanticPreservation:
    """The plan's required catches (§12) — each must reject."""

    @staticmethod
    def check(original, rewritten):
        from humanizer.validation import validate
        return validate(original, rewritten)

    def test_negation_flip_detected(self):
        r = self.check("The defendant was not found guilty.", "The defendant was found guilty.")
        assert not r.is_valid and r.negation_preserved is False

    def test_number_change_detected(self):
        r = self.check("Revenue was $4.2 billion.", "Revenue was $4.5 billion.")
        assert not r.is_valid and any("4.2" in e for e in r.errors)

    def test_entity_removal_detected(self):
        r = self.check("John threw rocks into the lake.", "Rocks were thrown into the lake.")
        assert not r.is_valid and "John" in r.missing_information

    def test_causal_inversion_detected(self):
        r = self.check("The crash caused the traffic jam.", "The traffic jam caused the crash.")
        assert not r.is_valid and any("reversed" in e for e in r.errors)

    def test_modality_strengthening_detected(self):
        r = self.check("The treatment might reduce symptoms.", "The treatment will reduce symptoms.")
        assert not r.is_valid and any("uncertainty" in e for e in r.errors)

    def test_unsupported_addition_detected(self):
        r = self.check("The study found a correlation.", "The landmark study found a strong causal link.")
        assert not r.is_valid
        assert {"landmark", "strong"} <= set(r.unsupported_additions)

    def test_event_change_detected(self):
        r = self.check("John threw rocks into the lake.", "John was hit by rocks near the lake.")
        assert not r.is_valid and r.suspected_meaning_changes

    def test_condition_flip_detected(self):
        r = self.check("Refunds are issued if the item is unused.", "Refunds are issued unless the item is unused.")
        assert not r.is_valid

    def test_comparison_reversal_detected(self):
        r = self.check("The new model uses more memory than the old one.",
                       "The new model uses less memory than the old one.")
        assert not r.is_valid

    def test_comparison_swapped_entities_detected(self):
        r = self.check("Model A is faster than Model B.", "Model B is faster than Model A.")
        assert not r.is_valid and any("comparison" in e for e in r.errors)

    def test_added_cause_detected(self):
        r = self.check("Sales rose. Prices fell.", "Sales rose because prices fell.")
        assert not r.is_valid and any("causal" in e for e in r.errors)

    def test_duplicated_word_is_grammar_problem(self):
        r = self.check("The team reviewed the plan.", "The the team reviewed the plan.")
        assert any("duplicated" in g for g in r.grammar_problems)


@needs_spacy
class TestValidRewritesAccepted:
    @pytest.mark.parametrize("original, rewritten", [
        ("John threw rocks into the lake.", "Rocks were thrown into the lake by John."),
        ("We met in order to discuss the end result.", "We met to discuss the result."),
        ("The new policy helps small businesses grow quickly.", "The new policy helps small firms expand rapidly."),
        ("The researchers did not observe any side effects in 2021.",
         "No side effects were observed by the researchers in 2021."),
        ("The plan was reviewed by the team.", "The team reviewed the plan."),
    ])
    def test_meaning_preserving_rewrite_accepted(self, original, rewritten):
        from humanizer.validation import validate
        r = validate(original, rewritten)
        assert r.is_valid, r.errors + r.warnings

    def test_identical_text_trivially_valid(self):
        from humanizer.validation import validate
        assert validate("Same text.", "Same text.").confidence == 1.0


@needs_spacy
def test_strictness_controls_major_issues():
    from humanizer.config import RewriterConfig
    from humanizer.validation import validate
    o, r = "Sales increased because prices fell.", "Sales increased and prices fell."  # one MAJOR issue
    assert validate(o, r, RewriterConfig(semantic_strictness="medium")).is_valid
    assert not validate(o, r, RewriterConfig(semantic_strictness="high")).is_valid


def test_report_has_required_sections():
    from humanizer.validation.base import CheckResult, Severity
    from humanizer.validation.validator import build_report
    c = CheckResult("facts")
    c.add("number", Severity.CRITICAL, "number '3' missing", "3", repair="Keep 3.")
    rep = build_report([c, CheckResult("additions_omissions")])
    assert rep.decision == "repair" and rep.repair_suggestions == ["Keep 3."]
    assert rep.checks["facts"]["passed"] is False and rep.missing_information == ["3"]
    assert 0 <= rep.confidence <= 1
