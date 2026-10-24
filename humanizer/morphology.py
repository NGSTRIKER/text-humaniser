"""Tiny English inflector used to fit a replacement word to the original's form.

Returns ``None`` when it cannot inflect safely — the caller must then
reject the candidate rather than guess.
"""

from __future__ import annotations

import re
from typing import Optional

VOWELS = "aeiou"

# base: (3sg, past, past participle, -ing)
IRREGULAR_VERBS: dict[str, tuple[str, str, str, str]] = {
    "be": ("is", "was", "been", "being"), "have": ("has", "had", "had", "having"),
    "do": ("does", "did", "done", "doing"), "go": ("goes", "went", "gone", "going"),
    "get": ("gets", "got", "gotten", "getting"), "make": ("makes", "made", "made", "making"),
    "take": ("takes", "took", "taken", "taking"), "give": ("gives", "gave", "given", "giving"),
    "see": ("sees", "saw", "seen", "seeing"), "come": ("comes", "came", "come", "coming"),
    "know": ("knows", "knew", "known", "knowing"), "think": ("thinks", "thought", "thought", "thinking"),
    "find": ("finds", "found", "found", "finding"), "tell": ("tells", "told", "told", "telling"),
    "become": ("becomes", "became", "become", "becoming"), "leave": ("leaves", "left", "left", "leaving"),
    "feel": ("feels", "felt", "felt", "feeling"), "bring": ("brings", "brought", "brought", "bringing"),
    "begin": ("begins", "began", "begun", "beginning"), "keep": ("keeps", "kept", "kept", "keeping"),
    "hold": ("holds", "held", "held", "holding"), "write": ("writes", "wrote", "written", "writing"),
    "stand": ("stands", "stood", "stood", "standing"), "hear": ("hears", "heard", "heard", "hearing"),
    "let": ("lets", "let", "let", "letting"), "mean": ("means", "meant", "meant", "meaning"),
    "set": ("sets", "set", "set", "setting"), "meet": ("meets", "met", "met", "meeting"),
    "run": ("runs", "ran", "run", "running"), "pay": ("pays", "paid", "paid", "paying"),
    "sit": ("sits", "sat", "sat", "sitting"), "speak": ("speaks", "spoke", "spoken", "speaking"),
    "lead": ("leads", "led", "led", "leading"), "grow": ("grows", "grew", "grown", "growing"),
    "lose": ("loses", "lost", "lost", "losing"), "fall": ("falls", "fell", "fallen", "falling"),
    "send": ("sends", "sent", "sent", "sending"), "build": ("builds", "built", "built", "building"),
    "understand": ("understands", "understood", "understood", "understanding"),
    "draw": ("draws", "drew", "drawn", "drawing"), "break": ("breaks", "broke", "broken", "breaking"),
    "spend": ("spends", "spent", "spent", "spending"), "cut": ("cuts", "cut", "cut", "cutting"),
    "rise": ("rises", "rose", "risen", "rising"), "drive": ("drives", "drove", "driven", "driving"),
    "buy": ("buys", "bought", "bought", "buying"), "wear": ("wears", "wore", "worn", "wearing"),
    "choose": ("chooses", "chose", "chosen", "choosing"), "seek": ("seeks", "sought", "sought", "seeking"),
    "throw": ("throws", "threw", "thrown", "throwing"), "catch": ("catches", "caught", "caught", "catching"),
    "deal": ("deals", "dealt", "dealt", "dealing"), "win": ("wins", "won", "won", "winning"),
    "show": ("shows", "showed", "shown", "showing"), "teach": ("teaches", "taught", "taught", "teaching"),
    "sell": ("sells", "sold", "sold", "selling"), "fight": ("fights", "fought", "fought", "fighting"),
    "put": ("puts", "put", "put", "putting"), "say": ("says", "said", "said", "saying"),
    "read": ("reads", "read", "read", "reading"), "lay": ("lays", "laid", "laid", "laying"),
    "forget": ("forgets", "forgot", "forgotten", "forgetting"), "hide": ("hides", "hid", "hidden", "hiding"),
    "shake": ("shakes", "shook", "shaken", "shaking"), "strike": ("strikes", "struck", "struck", "striking"),
    "light": ("lights", "lit", "lit", "lighting"), "bear": ("bears", "bore", "borne", "bearing"),
    "fly": ("flies", "flew", "flown", "flying"), "swim": ("swims", "swam", "swum", "swimming"),
    "eat": ("eats", "ate", "eaten", "eating"), "drink": ("drinks", "drank", "drunk", "drinking"),
    "sleep": ("sleeps", "slept", "slept", "sleeping"), "feed": ("feeds", "fed", "fed", "feeding"),
    "hit": ("hits", "hit", "hit", "hitting"), "shut": ("shuts", "shut", "shut", "shutting"),
    "spread": ("spreads", "spread", "spread", "spreading"), "split": ("splits", "split", "split", "splitting"),
}

IRREGULAR_PLURALS = {
    "child": "children", "person": "people", "man": "men", "woman": "women",
    "foot": "feet", "tooth": "teeth", "mouse": "mice", "goose": "geese",
    "analysis": "analyses", "crisis": "crises", "thesis": "theses", "datum": "data",
    "criterion": "criteria", "phenomenon": "phenomena", "life": "lives", "knife": "knives",
    "leaf": "leaves", "half": "halves", "wife": "wives",
}


# Longer verbs whose final consonant doubles (stress on the last syllable).
DOUBLING = frozenset(
    "stop plan drop ship shop chat grab plug skip slip trip step spot snap clap wrap drag "
    "stir scan swap knit admit commit permit prefer refer occur control regret submit "
    "equip omit transfer compel expel propel patrol".split()
)


def _is_cvc(word: str) -> bool:
    """Consonant-vowel-consonant ending that doubles (stop→stopped, but open→opened)."""
    if word in DOUBLING:
        return True
    return (
        len(word) == 3
        and word[-1] not in VOWELS + "wxy"
        and word[-2] in VOWELS
        and word[-3] not in VOWELS
    )


def plural(noun: str) -> str:
    if " " in noun:
        head, _, last = noun.rpartition(" ")
        return f"{head} {plural(last)}"
    if noun in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[noun]
    if re.search(r"[^aeiou]y$", noun):
        return noun[:-1] + "ies"
    if re.search(r"(ss|x|z|ch|sh|us|as|is)$", noun) or noun in {"lens", "bus", "gas"}:
        return noun + "es"
    if noun.endswith("s"):
        return noun
    return noun + "s"


def third_person(verb: str) -> str:
    if verb in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[verb][0]
    return plural(verb) if not verb.endswith("o") else verb + "es"


def past(verb: str, participle: bool = False) -> str:
    if verb in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[verb][2 if participle else 1]
    if verb.endswith("e"):
        return verb + "d"
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ied"
    if _is_cvc(verb):
        return verb + verb[-1] + "ed"
    return verb + "ed"


def gerund(verb: str) -> str:
    if verb in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[verb][3]
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith(("ee", "ye", "oe")):
        return verb[:-1] + "ing"
    if _is_cvc(verb):
        return verb + verb[-1] + "ing"
    return verb + "ing"


def inflect(lemma: str, tag: str, known_irregular: frozenset[str] = frozenset()) -> Optional[str]:
    """Put ``lemma`` into the form described by Penn ``tag``.

    ``known_irregular`` lists verb lemmas that WordNet says are irregular;
    if such a verb is missing from our table we refuse (return ``None``).
    """
    lemma = lemma.lower()
    if tag in {"NN", "NNP", "VB", "VBP", "JJ", "RB"}:
        return lemma
    if tag in {"NNS", "NNPS"}:
        return plural(lemma)
    if tag.startswith("VB"):
        if " " in lemma:
            head, _, rest = lemma.partition(" ")
            inner = inflect(head, tag, known_irregular)
            return f"{inner} {rest}" if inner else None
        if lemma in known_irregular and lemma not in IRREGULAR_VERBS:
            return None
        return {"VBZ": third_person, "VBG": gerund}.get(tag, lambda v: past(v, tag == "VBN"))(lemma)
    return None  # comparatives/superlatives etc.: too risky without a lexicon
