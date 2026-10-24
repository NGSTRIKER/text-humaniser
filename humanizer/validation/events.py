"""Event and semantic-role preservation (§12 B).

Each clause becomes ``(agent, predicate, patient)`` independent of voice:

* active  "John threw rocks"            → (john, throw, rock)
* passive "Rocks were thrown by John"   → (john, throw, rock)

Then we check that original events survive with the same participants
and that no participant switched sides (agent ↔ patient).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from humanizer.analyzer import DocumentAnalysis
from humanizer.validation.base import CheckResult, Pair, Severity, related, span_key

LIGHT_VERBS = {"be", "have", "do", "get", "make", "seem", "become"}
PRONOUNS_AND_PROFORMS = frozenset({
    "it", "this", "that", "these", "those", "they", "them", "there",
    "something", "anything", "everything", "nothing", "what", "which", "who", "whom",
    "event", "situation", "things",
})


@dataclass(frozen=True)
class Event:
    predicate: str
    agent: Optional[str]
    patient: Optional[str]
    negated: bool
    text: str


def extract_events(doc: DocumentAnalysis) -> list[Event]:
    from humanizer.validation.base import dispensable_ranges
    skip = dispensable_ranges(doc.text)
    events = []
    for s in doc.sentences:
        for c in s.clauses:
            if c.verb_lemma.lower() in LIGHT_VERBS:
                continue
            verb_tok = doc.tokens[c.verb_index] if 0 <= c.verb_index < len(doc.tokens) else None
            if verb_tok and any(a <= verb_tok.char_start < b for a, b in skip):
                continue
            if c.is_passive:
                agent, patient = span_key(c.agent, doc), span_key(c.subject, doc)
            else:
                agent, patient = span_key(c.subject, doc), span_key(c.direct_object, doc)
            events.append(Event(c.verb_lemma.lower(), agent, patient, c.negated, s.text))
    return events


def _same(a: Optional[str], b: Optional[str]) -> bool:
    return a is not None and b is not None and related(a, b, "n")


def check_events(pair: Pair) -> CheckResult:
    res = CheckResult("events")
    oe, re_ = extract_events(pair.original), extract_events(pair.rewritten)
    if not oe:
        res.skipped, res.note = True, "no content-verb events found"
        return res
    matched = 0
    for e in oe:
        cands = [x for x in re_ if related(e.predicate, x.predicate, "v")]
        if not cands:
            res.add("event", Severity.MINOR, f"event '{e.predicate}' not found (may be paraphrased)", e.text)
            continue
        best = max(cands, key=lambda x: (_same(e.agent, x.agent) + _same(e.patient, x.patient)))
        matched += 1
        if e.agent and e.patient and _same(e.agent, best.patient) and _same(e.patient, best.agent):
            res.add("role", Severity.CRITICAL,
                    f"roles reversed in '{e.predicate}': {e.agent} → {e.patient} became {best.agent} → {best.patient}",
                    e.text, best.text, repair=f"Keep '{e.agent}' as the one who does '{e.predicate}' to '{e.patient}'.")
        elif e.agent and best.agent is None and not _same(e.agent, best.patient):
            if e.agent.lower() in PRONOUNS_AND_PROFORMS:
                res.add("role", Severity.MINOR, f"agent pronoun '{e.agent}' of '{e.predicate}' dropped or altered", e.text, best.text)
            else:
                res.add("role", Severity.MAJOR, f"agent '{e.agent}' of '{e.predicate}' dropped", e.text, best.text,
                        repair=f"Say who did it: keep '{e.agent}' as the agent of '{e.predicate}'.")
        elif e.patient and best.patient and not _same(e.patient, best.patient) and not _same(e.patient, best.agent):
            if e.patient.lower() in PRONOUNS_AND_PROFORMS or best.patient.lower() in PRONOUNS_AND_PROFORMS:
                res.add("role", Severity.MINOR, f"object '{e.patient}' of '{e.predicate}' rephrased as {best.patient}", e.text, best.text)
            else:
                res.add("role", Severity.MAJOR, f"object of '{e.predicate}' changed: {e.patient} → {best.patient}",
                        e.text, best.text)
        if e.negated != best.negated:
            # MAJOR, not critical: negation may legitimately move from the verb to a
            # determiner ("did not see any" → "no … were seen").  A real flip also
            # changes the negation count, which the facts check treats as critical.
            res.add("polarity", Severity.MAJOR, f"'{e.predicate}' negation moved or changed", e.text, best.text,
                    repair=f"Keep '{e.predicate}' {'negated' if e.negated else 'affirmative'} as in the original.")

    # A participant who acted in the original but only receives an action now
    # (e.g. "John threw rocks" → "John was hit by rocks") is a role change.
    o_agents = {e.agent for e in oe if e.agent}
    r_agents = {e.agent for e in re_ if e.agent}
    r_patients = {e.patient for e in re_ if e.patient}
    for a in o_agents:
        if not any(_same(a, x) for x in r_agents) and any(_same(a, x) for x in r_patients):
            res.add("role", Severity.CRITICAL, f"'{a}' changed from doing the action to receiving it",
                    repair=f"Keep '{a}' as the one performing the action.")
    res.score = round(matched / len(oe), 3)
    return res
