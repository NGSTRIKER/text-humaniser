"""Rule engine (Stages 3–6): analyze → propose → gate → score → select → apply.

    >>> from humanizer.rule_engine import RuleEngine
    >>> result = RuleEngine().run("We met in order to discuss the end result.", seed=1)
    >>> result.text
    'We met to discuss the result.'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from humanizer.analyzer import analyze
from humanizer.candidates import Candidate, protected_spans
from humanizer.config import Intensity, RewriterConfig
from humanizer.edit_history import EditHistory
from humanizer.logging_config import get_logger
from humanizer.paragraph_flow import analyze_flow, swappable_pairs, swappable_paragraphs
from humanizer.schemas import EditRecord
from humanizer.transformations import Rule, RuleContext, default_rules
from humanizer.probability import make_rng
from humanizer.profiles import resolve_profile
from humanizer.weighting import gate, select

log = get_logger("rule_engine")


@dataclass
class RuleEngineResult:
    original_text: str
    text: str
    history: EditHistory
    analysis: Any = None

    @property
    def changed(self) -> bool:
        return self.text != self.original_text


class RuleEngine:
    """Runs all rules over a text and applies the selected edits."""

    def __init__(
        self,
        config: Optional[RewriterConfig] = None,
        rules: Optional[list[Rule]] = None,
        nlp: Any = None,
    ) -> None:
        self.config = config or RewriterConfig()
        self.rules = rules if rules is not None else default_rules(self.config)
        self.nlp = nlp

    def run(
        self,
        text: str,
        *,
        intensity: Optional[Intensity | str] = None,
        protected_terms: Iterable[str] = (),
        seed: Optional[int] = None,
        seed_key: object = 0,
    ) -> RuleEngineResult:
        """Rewrite ``text``.  ``seed_key`` (e.g. a chunk index) derives an
        independent but reproducible sub-seed from ``seed``."""
        intensity = Intensity(intensity) if intensity else self.config.intensity
        seed = seed if seed is not None else self.config.seed
        rng = make_rng(seed, seed_key)
        profile = resolve_profile(intensity, self.config)
        history = EditHistory(original_text=text)
        if not text.strip():
            return RuleEngineResult(text, text, history)

        analysis = analyze(text, nlp=self.nlp)
        ctx = RuleContext(text=text, analysis=analysis, intensity=intensity)
        protected = protected_spans(analysis, protected_terms)

        proposals: list[Candidate] = []
        for rule in self.rules:
            if not rule.active_for(intensity):
                continue
            try:
                proposals.extend(rule.propose(ctx))
            except Exception:  # one broken rule must not kill the run
                log.exception("Rule %s failed; skipping it", rule.rule_id)

        acceptable: list[Candidate] = []
        rejected: list[tuple[Candidate, str]] = []
        for c in proposals:
            if c.transform_type == "paragraph_flow":
                continue
            reason = gate(c, self.config, intensity, protected, profile)
            (rejected.append((c, reason)) if reason else acceptable.append(c))

        # Style edits (e.g. contraction expansion in academic mode) are
        # mechanical and meaning-safe: apply all of them, outside the budget.
        style_edits = [c for c in acceptable if c.transform_type == "style"]
        acceptable = [c for c in acceptable if c.transform_type != "style"]
        cap = profile.edit_cap(len(text.split()))
        selected, lost = select(
            acceptable, self.config, cap, rng,
            structural_limit=profile.structural_cap(len(analysis.sentences)),
            sentence_bounds=[(s.char_start, s.char_end) for s in analysis.sentences],
            profile=profile,
        )
        rejected.extend(lost)
        for c in style_edits:
            if any(c.char_start < o.char_end and o.char_start < c.char_end for o in selected):
                rejected.append((c, "overlaps a selected edit"))
            else:
                c.selection_probability = 1.0
                selected.append(c)

        n = 0
        for c, ok, why in [(c, True, None) for c in selected] + [(c, False, r) for c, r in rejected]:
            n += 1
            history.records.append(
                EditRecord(
                    edit_id=f"e{n}", original_span=c.original, replacement=c.replacement or "",
                    rule_id=c.rule_id, transform_type=c.transform_type, reason=c.reason,
                    confidence=c.confidence, semantic_risk=c.meaning_risk, accepted=ok,
                    char_start=c.char_start, char_end=c.char_end, score=c.score,
                    rejection_reason=why,
                    selection_probability=c.selection_probability if ok else None,
                )
            )
        history.records.sort(key=lambda r: (r.char_start, not r.accepted))
        for i, r in enumerate(history.records, 1):
            r.edit_id = f"e{i}"

        new_text = history.render()

        # Pass 2: Flow & macro-structure (sentence and paragraph reordering)
        reordering = getattr(self.config, "sentence_reordering", False) or any(
            getattr(r, "reordering", False) for r in self.rules
        )
        if reordering:
            analysis2 = analyze(new_text, nlp=self.nlp)
            flow2 = analyze_flow(analysis2)
            sents2 = analysis2.sentences

            flow_candidates: list[Candidate] = []

            # 1. Swappable sentence pairs
            pairs = swappable_pairs(flow2, allow_short=True)
            for x, y in pairs:
                s1, s2 = sents2[x], sents2[y]
                orig_span = new_text[s1.char_start : s2.char_end]
                trailing = orig_span[len(orig_span.rstrip()):]
                s1_clean = s1.text.rstrip()
                s2_clean = s2.text.rstrip()
                gap = new_text[s1.char_start + len(s1_clean) : s2.char_start]
                if not gap:
                    gap = " "
                replacement = s2_clean + gap + s1_clean + trailing
                flow_candidates.append(
                    Candidate(
                        rule_id="paragraph_flow.swap",
                        transform_type="paragraph_flow",
                        char_start=s1.char_start,
                        char_end=s2.char_end,
                        original=orig_span,
                        replacement=replacement,
                        reason="Reordered two logically independent sentences within the paragraph.",
                        confidence=0.7,
                        meaning_risk=0.3,
                        grammar_risk=0.12,
                        usefulness=0.55,
                        context_fit=0.7,
                        diversity_gain=0.6,
                        preserves_protected=True,
                        priority=2,
                    )
                )

            # 2. Swappable paragraphs
            p_pairs = swappable_paragraphs(flow2, analysis2)
            for p1, p2 in p_pairs:
                p1_start = sents2[flow2.paragraphs[p1][0]].char_start
                p1_end = sents2[flow2.paragraphs[p1][-1]].char_end
                p2_start = sents2[flow2.paragraphs[p2][0]].char_start
                p2_end = sents2[flow2.paragraphs[p2][-1]].char_end
                orig_span = new_text[p1_start:p2_end]
                trailing = orig_span[len(orig_span.rstrip()):]
                p1_text = new_text[p1_start:p1_end]
                p2_text = new_text[p2_start:p2_end]
                p1_clean = p1_text.rstrip()
                p2_clean = p2_text.rstrip()
                gap = new_text[p1_start + len(p1_clean) : p2_start]
                if not gap:
                    gap = "\n\n"
                replacement = p2_clean + gap + p1_clean + trailing
                flow_candidates.append(
                    Candidate(
                        rule_id="paragraph_flow.paragraph_swap",
                        transform_type="paragraph_flow",
                        char_start=p1_start,
                        char_end=p2_end,
                        original=orig_span,
                        replacement=replacement,
                        reason="Reordered two logically independent paragraphs to improve document flow.",
                        confidence=0.7,
                        meaning_risk=0.25,
                        grammar_risk=0.12,
                        usefulness=0.55,
                        context_fit=0.7,
                        diversity_gain=0.6,
                        preserves_protected=True,
                        priority=2,
                    )
                )

            selected_flow: list[Candidate] = []
            for fc in flow_candidates:
                if any(not (fc.char_end <= sf.char_start or sf.char_end <= fc.char_start) for sf in selected_flow):
                    continue
                fc.selection_probability = 1.0
                selected_flow.append(fc)

            for sf in selected_flow:
                n += 1
                history.records.append(
                    EditRecord(
                        edit_id=f"e{n}",
                        original_span=sf.original,
                        replacement=sf.replacement or "",
                        rule_id=sf.rule_id,
                        transform_type=sf.transform_type,
                        reason=sf.reason,
                        confidence=sf.confidence,
                        semantic_risk=sf.meaning_risk,
                        accepted=True,
                        char_start=sf.char_start,
                        char_end=sf.char_end,
                        score=sf.score,
                        rejection_reason=None,
                        selection_probability=sf.selection_probability,
                    )
                )

            if selected_flow:
                history.records.sort(key=lambda r: (r.transform_type == "paragraph_flow", r.char_start, not r.accepted))
                for i, r in enumerate(history.records, 1):
                    r.edit_id = f"e{i}"
                new_text = history.render()
                analysis = analyze(new_text, nlp=self.nlp)

        log.info(
            "Rule engine: %d proposed, %d applied, %d rejected (cap %d, %s)",
            len(proposals), len(selected), len(rejected), cap, intensity.value,
        )
        return RuleEngineResult(text, new_text, history, analysis)
