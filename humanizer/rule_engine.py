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
        log.info(
            "Rule engine: %d proposed, %d applied, %d rejected (cap %d, %s)",
            len(proposals), len(selected), len(rejected), cap, intensity.value,
        )
        return RuleEngineResult(text, new_text, history, analysis)
