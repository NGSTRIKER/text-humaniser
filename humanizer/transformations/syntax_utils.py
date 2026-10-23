"""Helpers shared by the structural transformations (Stage 5)."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable, Optional

from humanizer.analyzer import load_nlp

SUBJ_TO_OBJ = {"i": "me", "he": "him", "she": "her", "we": "us", "they": "them", "who": "whom"}
OBJ_TO_SUBJ = {"me": "I", "him": "he", "her": "she", "us": "we", "them": "they", "whom": "who"}
FIRST_SECOND_PERSON = frozenset({"i", "we", "you", "me", "us"})
PLURAL_PRONOUNS = frozenset({"we", "they", "you", "these", "those", "us", "them"})

CONTRACTIONS = {
    "is": "isn't", "are": "aren't", "was": "wasn't", "were": "weren't", "has": "hasn't",
    "have": "haven't", "had": "hadn't", "will": "won't", "can": "can't", "could": "couldn't",
    "would": "wouldn't", "should": "shouldn't", "must": "mustn't", "does": "doesn't",
    "do": "don't", "did": "didn't",
}


@lru_cache(maxsize=2048)
def parse(text: str) -> Any:
    """Parse a sentence (cached — rules often re-parse the same text)."""
    return load_nlp()(text)


def span_text(tokens: Iterable[Any]) -> str:
    toks = sorted(tokens, key=lambda t: t.i)
    return "".join(t.text_with_ws for t in toks).strip()


def subtree(tok: Any) -> list[Any]:
    return list(tok.subtree)


def tidy(text: str) -> str:
    """Collapse spaces and remove spaces before punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+(n't)\b", r"\1", text)
    return text


def capitalize_first(text: str) -> str:
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1 :]
    return text


def decapitalize(tokens: list[Any], text: str) -> str:
    """Lower-case the first letter unless it starts a name or 'I'."""
    if not tokens:
        return text
    first = min(tokens, key=lambda t: t.i)
    if first.pos_ == "PROPN" or first.text == "I" or first.ent_type_ or (first.text.isupper() and len(first.text) > 1):
        return text
    return text[:1].lower() + text[1:]


def is_plural(head: Any) -> bool:
    if head.lower_ in PLURAL_PRONOUNS:
        return True
    if head.tag_ in {"NNS", "NNPS"}:
        return True
    return any(c.dep_ == "conj" for c in head.children)


def be_form(tense: str, subj_head: Any, subj_text: str) -> str:
    low = subj_text.lower()
    if tense == "past":
        return "were" if (is_plural(subj_head) or low == "you") and low != "i" else "was"
    if low == "i":
        return "am"
    return "are" if is_plural(subj_head) or low == "you" else "is"


def have_form(tense: str, subj_head: Any, subj_text: str) -> str:
    if tense == "past":
        return "had"
    low = subj_text.lower()
    return "have" if is_plural(subj_head) or low in {"i", "you"} else "has"


def do_form(tense: str, subj_head: Any, subj_text: str) -> str:
    if tense == "past":
        return "did"
    low = subj_text.lower()
    return "do" if is_plural(subj_head) or low in {"i", "you"} else "does"


def add_negation(words: list[str], neg: Optional[Any]) -> list[str]:
    """Insert 'not' (or contract with "n't") after the first auxiliary."""
    if neg is None or not words:
        return words
    first = words[0]
    if neg.text.lower() == "n't" and first.lower() in CONTRACTIONS:
        return [CONTRACTIONS[first.lower()]] + words[1:]
    return [first, neg.text if neg.text.lower() != "n't" else "not"] + words[1:]


def final_punct(doc: Any) -> str:
    if not len(doc):
        return ""
    last = doc[-1]
    return last.text if last.is_punct and last.text in {".", "!", "?"} else ""


def pronoun_swap(tokens: list[Any], mapping: dict[str, str], text: str) -> str:
    if len(tokens) == 1 and tokens[0].lower_ in mapping:
        out = mapping[tokens[0].lower_]
        return out
    return text
