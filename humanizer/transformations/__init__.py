"""Rule modules. ``default_rules()`` returns the Stage 3 rule set."""

from __future__ import annotations

from humanizer.transformations.base import Rule, RuleContext
from humanizer.transformations.phrases import NominalizationRule
from humanizer.transformations.redundancy import RedundantPhraseRule
from humanizer.transformations.repetition import (
    DoubledWordRule,
    RepeatedOpeningRule,
    RepeatedVocabularyRule,
)
from humanizer.transformations.sentence_structure import SentenceStructureRule
from humanizer.transformations.synonyms import SynonymRule
from humanizer.transformations.voice import VoiceRule
from humanizer.transformations.word_order import PhraseOrderRule
from humanizer.transformations.wordiness import WordinessRule


from humanizer.styles import get_profile
from humanizer.transformations.contractions import ContractionRule


def default_rules(config: object | None = None) -> list[Rule]:
    """All rules (Stages 3-5).  ``config`` toggles optional ones."""
    rules: list[Rule] = [
        RedundantPhraseRule(),
        WordinessRule(),
        DoubledWordRule(),
        RepeatedVocabularyRule(),
        RepeatedOpeningRule(),
        NominalizationRule(),
        VoiceRule(getattr(config, "voice_mode", "selective")),
        PhraseOrderRule(),
        SentenceStructureRule(reordering=getattr(config, "sentence_reordering", False)),
        ContractionRule(enabled=get_profile(getattr(config, "style", "neutral")).expand_contractions),
    ]
    if getattr(config, "synonyms_enabled", True):
        embeddings = None
        if getattr(config, "use_embeddings", False):
            from humanizer.context_analyzer import EmbeddingScorer

            embeddings = EmbeddingScorer(getattr(config, "embedding_model", "en_core_web_md"))
        rules.append(
            SynonymRule(
                embeddings=embeddings,
                min_sense_confidence=getattr(config, "min_sense_confidence", 0.3),
                style=getattr(config, "style", "neutral"),
            )
        )
    return rules


__all__ = ["Rule", "RuleContext", "default_rules"]
