"""Active ↔ passive voice conversion (Stage 5).

Works on a single simple clause and keeps tense, aspect, modality,
negation, agreement, pronoun case and the participants:

    John threw rocks into the lake.  ⇄  Rocks were thrown into the lake by John.
    John did not throw the ball.     →  The ball was not thrown by John.
    John might throw the ball.       →  The ball might be thrown by John.

Anything it cannot convert safely returns ``None`` with a reason, e.g.
agentless passives ("Mistakes were made."), ditransitives, questions,
stative verbs ("She has a car."), or multi-clause sentences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from humanizer.candidates import Candidate
from humanizer.config import Intensity
from humanizer.morphology import IRREGULAR_VERBS, gerund, past, third_person
from humanizer.transformations.base import Rule, RuleContext
from humanizer.transformations.syntax_utils import (
    FIRST_SECOND_PERSON,
    OBJ_TO_SUBJ,
    SUBJ_TO_OBJ,
    add_negation,
    be_form,
    capitalize_first,
    decapitalize,
    do_form,
    final_punct,
    have_form,
    is_plural,
    parse,
    pronoun_swap,
    span_text,
    subtree,
    tidy,
)

# Verbs that describe states and do not passivise naturally.
STATIVE = frozenset(
    "have lack resemble cost weigh fit suit possess become mean contain equal "
    "seem appear own deserve marry".split()
)
REFLEXIVE = frozenset({"myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves"})
BLOCKING_DEPS = frozenset(
    {"dative", "ccomp", "xcomp", "conj", "csubj", "expl", "oprd", "attr", "acomp", "parataxis"}
)
POST_DEPS_OK = frozenset({"prep", "advmod", "npadvmod", "punct", "prt", "advcl"})


@dataclass(frozen=True)
class VoiceResult:
    text: Optional[str]
    note: str


def _irregular_unknown(lemma: str) -> bool:
    """WordNet says irregular but our table lacks it → refuse to guess."""
    try:
        from humanizer.word_graph import get_graph

        exc = get_graph().wn._exception_map["v"]
        irregular = {v for forms in exc.values() for v in forms}
    except Exception:
        irregular = set()
    return lemma in irregular and lemma not in IRREGULAR_VERBS


def _single_root(doc: Any) -> Optional[Any]:
    sents = list(doc.sents)
    if len(sents) != 1:
        return None
    return sents[0].root


def to_passive(sentence: str) -> VoiceResult:
    """Convert an active sentence to passive, or explain why not."""
    doc = parse(sentence.strip())
    root = _single_root(doc)
    if root is None or root.pos_ != "VERB":
        return VoiceResult(None, "not a single clause with a main verb")
    if sentence.strip().endswith("?"):
        return VoiceResult(None, "question")
    kids = list(root.children)
    subjs = [c for c in kids if c.dep_ == "nsubj"]
    objs = [c for c in kids if c.dep_ in {"dobj", "obj"}]
    if len(subjs) != 1 or len(objs) != 1:
        return VoiceResult(None, "needs exactly one subject and one direct object")
    # "to the client" is parsed as dative but is just a prepositional phrase
    to_phrase = lambda c: c.dep_ == "dative" and c.pos_ == "ADP"  # noqa: E731
    if any(c.dep_ in BLOCKING_DEPS | {"nsubjpass", "auxpass"} and not to_phrase(c) for c in kids):
        return VoiceResult(None, "clause structure too complex to convert safely")
    subj, obj = subjs[0], objs[0]
    if root.lemma_.lower() in STATIVE:
        return VoiceResult(None, f"stative verb {root.lemma_!r} does not passivise naturally")
    if obj.lower_ in REFLEXIVE or subj.lower_ in {"there", "it"} and len(subtree(subj)) == 1 and subj.lower_ == "there":
        return VoiceResult(None, "reflexive or expletive")
    if any(c.dep_ not in POST_DEPS_OK and not to_phrase(c) for c in kids if c.i > root.i and c is not obj):
        return VoiceResult(None, "unsupported material after the verb")

    auxes = [c for c in kids if c.dep_ == "aux"]
    neg = next((c for c in kids if c.dep_ == "neg"), None)
    modal = next((a for a in auxes if a.tag_ == "MD"), None)
    have = next((a for a in auxes if a.lemma_.lower() == "have"), None)
    be = next((a for a in auxes if a.lemma_.lower() == "be"), None)
    do = next((a for a in auxes if a.lemma_.lower() == "do"), None)
    if any(a not in (modal, have, be, do) for a in auxes):
        return VoiceResult(None, "unsupported auxiliary")
    progressive = be is not None and root.tag_ == "VBG"
    if (modal or have) and progressive:
        return VoiceResult(None, "perfect/modal progressive passive sounds awkward")
    if root.tag_ == "VB" and not (modal or do):
        return VoiceResult(None, "imperative or infinitive")
    if root.tag_ == "VBN" and not have:
        return VoiceResult(None, "unexpected participle")

    if modal:
        tense = None
    elif have:
        tense = "past" if have.tag_ == "VBD" else "present"
    elif be:
        tense = "past" if be.tag_ == "VBD" else "present"
    elif do:
        tense = "past" if do.tag_ == "VBD" else "present"
    else:
        tense = "past" if root.tag_ == "VBD" else "present"

    lemma = root.lemma_.lower()
    if _irregular_unknown(lemma):
        return VoiceResult(None, f"irregular verb {lemma!r} not in the inflection table")
    participle = past(lemma, participle=True)

    obj_toks, subj_toks = subtree(obj), subtree(subj)
    new_subj = pronoun_swap(obj_toks, OBJ_TO_SUBJ, span_text(obj_toks))
    agent = pronoun_swap(subj_toks, SUBJ_TO_OBJ, span_text(subj_toks))
    if agent == span_text(subj_toks):
        agent = decapitalize(subj_toks, agent)

    if modal:
        chain = [modal.text, "have", "been"] if have else [modal.text, "be"]
    elif have:
        chain = [have_form(tense, obj, new_subj), "been"]
    elif progressive:
        chain = [be_form(tense, obj, new_subj), "being"]
    else:
        chain = [be_form(tense, obj, new_subj)]
    chain = add_negation(chain, neg)

    subj_start = min(t.i for t in subj_toks)
    pre = [t for c in kids if c.i < subj_start for t in subtree(c)]
    mid = [c for c in kids if subj_start < c.i < root.i and c.dep_ in {"advmod", "npadvmod"}]
    prt = [c for c in kids if c.dep_ == "prt"]
    post = [c for c in kids if c.i > root.i and c is not obj and c.dep_ not in {"punct", "prt"}]

    parts: list[str] = []
    if pre:
        parts.append(span_text(pre))
    parts.append(new_subj if pre else capitalize_first(new_subj))
    parts += chain
    parts += [span_text(subtree(m)) for m in mid]
    parts.append(participle)
    parts += [p.text for p in prt]
    parts += [span_text(subtree(p)) for p in post]
    parts += ["by", agent]
    out = tidy(" ".join(parts) + final_punct(doc))
    return _verify(out, passive=True)


def to_active(sentence: str) -> VoiceResult:
    """Convert a passive sentence WITH an explicit agent to active."""
    doc = parse(sentence.strip())
    root = _single_root(doc)
    if root is None or root.pos_ != "VERB" or root.tag_ != "VBN":
        return VoiceResult(None, "not a single passive clause")
    if sentence.strip().endswith("?"):
        return VoiceResult(None, "question")
    kids = list(root.children)
    subj = next((c for c in kids if c.dep_ == "nsubjpass"), None)
    agent_prep = next((c for c in kids if c.dep_ == "agent"), None)
    if subj is None or not any(c.dep_ == "auxpass" for c in kids):
        return VoiceResult(None, "not passive")
    if agent_prep is None:
        return VoiceResult(None, "no agent ('by …'); the actor is unknown or intentionally omitted")
    pobj = next((c for c in agent_prep.children if c.dep_ == "pobj"), None)
    if pobj is None:
        return VoiceResult(None, "agent phrase has no noun")
    if any(c.dep_ in BLOCKING_DEPS for c in kids):
        return VoiceResult(None, "clause structure too complex to convert safely")

    auxes = [c for c in kids if c.dep_ in {"aux", "auxpass"}]
    neg = next((c for c in kids if c.dep_ == "neg"), None)
    modal = next((a for a in auxes if a.tag_ == "MD"), None)
    have = next((a for a in auxes if a.lemma_.lower() == "have"), None)
    being = any(a.lower_ == "being" for a in auxes)
    finite_be = next((a for a in auxes if a.lemma_.lower() == "be" and a.tag_ in {"VBZ", "VBP", "VBD"}), None)
    if modal and being:
        return VoiceResult(None, "modal progressive passive")

    lemma = root.lemma_.lower()
    if _irregular_unknown(lemma):
        return VoiceResult(None, f"irregular verb {lemma!r} not in the inflection table")
    pp = root.text.lower()

    agent_toks, subj_toks = subtree(pobj), subtree(subj)
    new_subj = pronoun_swap(agent_toks, OBJ_TO_SUBJ, span_text(agent_toks))
    new_obj = pronoun_swap(subj_toks, SUBJ_TO_OBJ, span_text(subj_toks))
    if new_obj == span_text(subj_toks):
        new_obj = decapitalize(subj_toks, new_obj)

    if modal:
        tense = None
    elif have:
        tense = "past" if have.tag_ == "VBD" else "present"
    elif finite_be is not None:
        tense = "past" if finite_be.tag_ == "VBD" else "present"
    else:
        return VoiceResult(None, "could not determine tense")

    singular_3rd = not is_plural(pobj) and new_subj.lower() not in {"i", "you"}
    if modal:
        verbs = [modal.text] + (["have", pp] if have else [lemma])
        verbs = add_negation(verbs, neg)
    elif have:
        verbs = [have_form(tense, pobj, new_subj)] + (["been", gerund(lemma)] if being else [pp])
        verbs = add_negation(verbs, neg)
    elif being:
        verbs = add_negation([be_form(tense, pobj, new_subj), gerund(lemma)], neg)
    elif neg is not None:
        verbs = add_negation([do_form(tense, pobj, new_subj), lemma], neg)
    else:
        if tense == "past":
            verbs = [past(lemma)]
        else:
            verbs = [third_person(lemma) if singular_3rd else lemma]

    subj_start = min(t.i for t in subj_toks)
    pre = [t for c in kids if c.i < subj_start for t in subtree(c)]
    mid = [c for c in kids if subj_start < c.i < root.i and c.dep_ in {"advmod", "npadvmod"}]
    prt = [c for c in kids if c.dep_ == "prt"]
    post = [c for c in kids if c.i > root.i and c is not agent_prep and c.dep_ not in {"punct", "prt"}]

    parts: list[str] = []
    if pre:
        parts.append(span_text(pre))
    parts.append(new_subj if pre else capitalize_first(new_subj))
    if mid and len(verbs) > 1:
        verbs = verbs[:-1] + [span_text(subtree(m)) for m in mid] + verbs[-1:]
    elif mid:
        parts += [span_text(subtree(m)) for m in mid]
    parts += verbs
    parts += [p.text for p in prt]
    parts.append(new_obj)
    parts += [span_text(subtree(p)) for p in post]
    out = tidy(" ".join(parts) + final_punct(doc))
    return _verify(out, passive=False)


def _verify(text: str, passive: bool) -> VoiceResult:
    """Re-parse the output and confirm it has the intended structure."""
    doc = parse(text)
    root = _single_root(doc)
    if root is None:
        return VoiceResult(None, "output did not parse as one sentence")
    deps = {c.dep_ for c in root.children}
    if passive and not ({"auxpass", "nsubjpass"} & deps):
        return VoiceResult(None, "output did not parse as passive")
    if not passive and "nsubj" not in deps:
        return VoiceResult(None, "output did not parse as active")
    return VoiceResult(text, "ok")


def should_convert(sentence: str) -> bool:
    """Is a voice change both possible and advisable for this sentence?

    Declines conversational first/second-person sentences and passives
    without an agent.
    """
    doc = parse(sentence.strip())
    root = _single_root(doc)
    if root is None:
        return False
    deps = {c.dep_: c for c in root.children}
    if "nsubjpass" in deps:
        return to_active(sentence).text is not None
    subj = deps.get("nsubj")
    if subj is not None and subj.lower_ in FIRST_SECOND_PERSON:
        return False
    if any(t.lower_ in {"n't", "'re", "'m", "'ll"} for t in doc):
        return False
    return to_passive(sentence).text is not None


class VoiceRule(Rule):
    """Selective voice conversion.

    * ``selective`` (default): passive→active when the agent is explicit;
      active→passive only when the object is already the topic (it was
      mentioned in the previous sentence), so the passive improves flow.
    * ``frequent``: also offers active→passive for any convertible clause.
    * ``disabled``: does nothing.
    """

    rule_id = "voice"
    description = "Convert between active and passive voice when it preserves meaning."
    transform_type = "voice"
    min_intensity = Intensity.MODERATE

    def __init__(self, mode: str = "selective") -> None:
        self.mode = mode

    def propose(self, ctx: RuleContext) -> list[Candidate]:
        if self.mode == "disabled":
            return []
        out: list[Candidate] = []
        sents = ctx.analysis.sentences
        for i, sent in enumerate(sents):
            if sent.voice == "passive":
                res = to_active(sent.text)
                if res.text:
                    out.append(self._cand(sent, res.text, "passive→active", 0.12,
                                          "Changed passive voice to active voice (agent is explicit)."))
                continue
            if sent.voice != "active" or not should_convert(sent.text):
                continue
            doc = parse(sent.text)
            obj = next((c for c in doc[:].root.children if c.dep_ in {"dobj", "obj"}), None)
            topical = False
            if obj is not None and i > 0:
                prev_lemmas = {
                    ctx.analysis.tokens[t].lemma.lower() for t in sents[i - 1].token_indices
                }
                topical = obj.lemma_.lower() in prev_lemmas
            is_consecutive_active = i > 0 and sents[i - 1].voice == "active"
            allow_passive = (
                self.mode == "frequent"
                or topical
                or ctx.intensity == Intensity.HEAVY
            )
            if allow_passive:
                res = to_passive(sent.text)
                if res.text:
                    why = ("object continues the previous sentence's topic" if topical
                           else "voice variety" if ctx.intensity == Intensity.HEAVY
                           else "voice variety to break consecutive active sentences" if is_consecutive_active
                           else "voice variety")
                    out.append(self._cand(sent, res.text, "active→passive",
                                          0.15 if topical else 0.18,
                                          f"Changed active voice to passive voice ({why})."))
        return out

    def _cand(self, sent: Any, new: str, direction: str, risk: float, reason: str) -> Candidate:
        return Candidate(
            rule_id=f"voice.{direction.replace('→', '_to_')}",
            transform_type=self.transform_type,
            char_start=sent.char_start, char_end=sent.char_end,
            original=sent.text, replacement=new, reason=reason,
            confidence=0.75, meaning_risk=risk, grammar_risk=0.15,
            usefulness=0.6, context_fit=0.7, diversity_gain=0.6,
            preserves_protected=True, priority=2,
        )
