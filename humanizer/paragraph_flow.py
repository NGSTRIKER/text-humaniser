"""Conservative paragraph-flow analysis (Stage 5).

Looks at each paragraph as a connected unit and records, for every
sentence, whether it *depends* on the one before it — even when it is
grammatically complete.  Example: "She approved it." is a full sentence
but cannot be moved, because "She" and "it" point backwards.

Used to:
* block sentence combining/reordering that would break references;
* (optionally, off by default) offer swaps of adjacent sentences that are
  logically independent.

Debug:  python -m humanizer.paragraph_flow "Some text. More text."
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

from humanizer.analyzer import DocumentAnalysis, analyze

CHRONOLOGY = ("then", "next", "later", "afterwards", "afterward", "finally", "first", "second",
              "third", "lastly", "after that", "before that", "meanwhile", "subsequently", "eventually")
BACK_REFERENCE_DETS = frozenset({"this", "that", "these", "those", "such"})
CONCLUSION_MARKERS = ("in conclusion", "overall", "in summary", "to summarize", "to sum up", "ultimately")


@dataclass
class SentenceFlow:
    index: int
    paragraph: int
    depends_on_previous: bool
    reasons: list[str] = field(default_factory=list)
    topic_overlap: float = 0.0     # share of content lemmas shared with previous sentence
    role: str = "body"             # intro | body | conclusion | single

    @property
    def logically_independent(self) -> bool:
        return not self.depends_on_previous


@dataclass
class FlowReport:
    paragraphs: list[list[int]]
    sentences: list[SentenceFlow]
    has_chronology: dict[int, bool]

    def summary(self) -> str:
        lines = []
        for p, idxs in enumerate(self.paragraphs):
            lines.append(f"Paragraph {p + 1}{' (chronological)' if self.has_chronology[p] else ''}:")
            for i in idxs:
                s = self.sentences[i]
                dep = "depends on previous: " + "; ".join(s.reasons) if s.depends_on_previous else "independent"
                lines.append(f"  [{i}] {s.role:<10} overlap={s.topic_overlap:.2f}  {dep}")
        return "\n".join(lines)


def paragraph_of_sentences(analysis: DocumentAnalysis) -> list[int]:
    """Paragraph number for every sentence (blank lines separate paragraphs)."""
    breaks = [m.end() for m in re.finditer(r"\n\s*\n", analysis.text)]
    out = []
    for s in analysis.sentences:
        out.append(sum(1 for b in breaks if b <= s.char_start))
    return out


def analyze_flow(analysis: DocumentAnalysis) -> FlowReport:
    para_of = paragraph_of_sentences(analysis)
    paragraphs: dict[int, list[int]] = {}
    for i, p in enumerate(para_of):
        paragraphs.setdefault(p, []).append(i)

    def content(si: int) -> set[str]:
        return {
            analysis.tokens[t].lemma.lower()
            for t in analysis.sentences[si].token_indices
            if analysis.tokens[t].pos in {"NOUN", "PROPN", "VERB", "ADJ"} and not analysis.tokens[t].is_stop
        }

    flows: list[SentenceFlow] = []
    chrono: dict[int, bool] = {}
    for p, idxs in paragraphs.items():
        chrono[p] = False
        for pos, si in enumerate(idxs):
            sent = analysis.sentences[si]
            low = sent.text.lower()
            reasons: list[str] = []
            first_in_para = pos == 0
            # discourse markers at the start
            for m in sent.discourse_markers:
                if m.char_start - sent.char_start <= 1:
                    reasons.append(f"starts with discourse marker {m.text!r}")
            # chronology
            for word in CHRONOLOGY:
                if re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", low):
                    chrono[p] = True
                    reasons.append(f"chronology marker {word!r}")
                    break
            # pronouns pointing back
            prev_nps = (
                {np_.text for np_ in analysis.sentences[idxs[pos - 1]].noun_phrases}
                if pos > 0 else set()
            )
            for pr in analysis.pronouns:
                if pr.sentence_index == si and (set(pr.candidates) & prev_nps or not pr.candidates):
                    reasons.append(f"pronoun {pr.text!r} refers back")
                    break
            # demonstratives / definite back-references
            if sent.token_indices:
                first_tok = analysis.tokens[sent.token_indices[0]]
                if first_tok.text.lower() in BACK_REFERENCE_DETS:
                    reasons.append(f"demonstrative {first_tok.text!r}")
            if any(re.search(r"(?<!\w)" + w + r"(?!\w)", low) for w in ("also", "too", "as well", "another", "the same")):
                reasons.append("additive/comparative reference")
            overlap = 0.0
            if not first_in_para:
                prev, cur = content(idxs[pos - 1]), content(si)
                overlap = len(prev & cur) / max(1, len(cur))
            if first_in_para:
                reasons = [r for r in reasons if not r.startswith("pronoun")]  # nothing before it in paragraph
            role = (
                "single" if len(idxs) == 1
                else "intro" if first_in_para
                else "conclusion" if pos == len(idxs) - 1 or any(low.startswith(c) for c in CONCLUSION_MARKERS)
                else "body"
            )
            flows.append(
                SentenceFlow(si, p, bool(reasons) and not first_in_para, reasons, round(overlap, 2), role)
            )
    flows.sort(key=lambda f: f.index)
    return FlowReport([paragraphs[p] for p in sorted(paragraphs)], flows, chrono)


def swappable_pairs(report: FlowReport) -> list[tuple[int, int]]:
    """Adjacent sentence pairs (i, i+1) that can be swapped without breaking flow.

    Very conservative: paragraph must have ≥4 sentences and no chronology;
    intro and conclusion stay in place; i, i+1 and i+2 must all be
    logically independent.
    """
    pairs = []
    for p, idxs in enumerate(report.paragraphs):
        if len(idxs) < 4 or report.has_chronology[p]:
            continue
        for k in range(1, len(idxs) - 2):
            a, b = idxs[k], idxs[k + 1]
            after = idxs[k + 2] if k + 2 < len(idxs) else None
            sa, sb = report.sentences[a], report.sentences[b]
            if sa.depends_on_previous or sb.depends_on_previous:
                continue
            if sa.role != "body" or sb.role != "body":
                continue
            if after is not None and report.sentences[after].depends_on_previous:
                continue
            pairs.append((a, b))
    return pairs


def _main(argv: list[str]) -> int:
    text = " ".join(argv) if argv else sys.stdin.read()
    print(analyze_flow(analyze(text)).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
