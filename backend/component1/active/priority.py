"""Which question is most worth asking now? Priority = chance it is unresolved x impact x urgency.

    signals = question_signals(state, events, risk_bundle, days_left=20, skip={"evaluation_metric"})

Three kinds of question compete for Buddy's one question slot (agent/policy.py):
    GAP      a missing field that can be asked about now (gaps.ready_gaps decides which)   chance 1.0
    CONFIRM  a field the conflict-risk model expects to be disputed or changed soon        chance = risk
    CLARIFY  a field whose value is too vague to build on (specific.py)                    chance 0.7
impact   = the field's importance weight (schema.BASE_WEIGHT) / 10 + 0.1 per field that depends on it
urgency  = 1, plus up to 1 more as the deadline gets within 30 days (fields needed to start building
           get the full amount, earlier fields 0.3 of it); a missing timeline gets x1.5 once half the plan is settled
decisive facts (sensitivity.py): when a missing or unsure fact decides whether the plan is feasible, its
gap or confirm question counts x1.5 and says why ("whether the plan is feasible depends on the dataset size...").
The numbers are hand-set starting values; sim_policies.py compares this order with the old fixed one.
Tried and removed: "confirm first" (boost a CONFIRM question while the fields that depend on it are still
open, hold back a GAP whose parent is at risk) did not reduce rework in the simulation
(results/policy_confirm_first_experiment.json).
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

from ..agent.signals import CLARIFY, CONFIRM, GAP, Signal
from ..common.schema import BASE_WEIGHT, FieldName
from ..decisions.domino import descendants
from ..gaps.gaps import ready_gaps
from .risk import field_risks
from .sensitivity import feasibility_dependencies
from .specific import vague_fields

CONFIRM_MIN_RISK = 0.20      # about twice the average chance of a change (10%) in the simulated teams
CLARIFY_CHANCE = 0.7
DECISIVE_BONUS = 0.5         # a question about a fact that decides feasibility counts (1 + this) as much
UNSURE_DECISIVE_CHANCE = 0.5  # an unsure decisive fact the risk model does not flag is still worth confirming
BUILD_FIELDS = {"algorithm", "evaluation_metric", "resources", "dataset_size", "timeline"}
DEADLINE_WINDOW_DAYS = 30


def enabled() -> bool:
    """C1_ACTIVE=off in backend/.env goes back to the old fixed order of gap questions."""
    return (os.environ.get("C1_ACTIVE") or "on").strip().lower() not in ("off", "0", "false", "no")


def impact(field: str) -> float:
    return BASE_WEIGHT[FieldName(field)] / 10 + 0.1 * len(descendants(field))


def urgency(field: str, days_left: Optional[float], settled_share: float) -> float:
    u = 1.0
    if days_left is not None:
        pressure = min(1.0, max(0.0, (DEADLINE_WINDOW_DAYS - days_left) / DEADLINE_WINDOW_DAYS))
        u += pressure * (1.0 if field in BUILD_FIELDS else 0.3)
    if field == "timeline" and days_left is None and settled_share >= 0.5:
        u *= 1.5
    return u


def _settled_share(state) -> float:
    from ..common.schema import FieldStatus
    return sum(1 for f in state.fields.values() if f.status == FieldStatus.CONFIRMED) / len(FieldName)


def question_signals(state, events: Iterable, risk_bundle: Optional[dict], days_left: Optional[float] = None,
                     skip: Iterable[str] = (), with_reasons: bool = True,
                     feasibility_bundle: Optional[dict] = None) -> list[Signal]:
    """GAP, CONFIRM and CLARIFY signals with their priority, highest first.

    skip: fields Buddy is already warning about (a failing plan check or a Domino re-check), so they
    do not get a second question. feasibility_bundle: the feasibility model, to find decisive facts
    (None: not looked for)."""
    skip = set(skip)
    share = _settled_share(state)
    out: list[Signal] = []

    def score(field: str, chance: float) -> float:
        return round(chance * impact(field) * urgency(field, days_left, share), 4)

    events = list(events)
    risks = field_risks(risk_bundle, state, events, days_left, with_reasons)
    decisive = {d["field"]: d for d in feasibility_dependencies(feasibility_bundle, state, events)}
    for name in ready_gaps(state):
        f = name.value
        n = len(descendants(f))
        why = f"not decided yet; {n} other field{'s' if n != 1 else ''} depend on it" if n else "not decided yet"
        chance = 1.0
        if f in decisive:
            why, chance = decisive[f]["text"], 1.0 + DECISIVE_BONUS
        out.append(Signal(GAP, f, detail=why, priority=score(f, chance)))

    vague = {f: q for f, q in vague_fields(state).items() if f not in skip}
    for f, question in vague.items():
        value = state.fields[FieldName(f)].value or ""
        out.append(Signal(CLARIFY, f, (value,), detail=question, priority=score(f, CLARIFY_CHANCE)))

    for f, r in risks.items():
        unsure_decisive = f in decisive and decisive[f]["known"] == "unsure"
        if f in skip or f in vague or (r["risk"] < CONFIRM_MIN_RISK and not unsure_decisive):
            continue
        value = state.fields[FieldName(f)].value or ""
        reasons, chance = list(r["reasons"]), r["risk"]
        if unsure_decisive:
            reasons = ["whether the plan is feasible depends on it", *reasons]
            chance = max(chance, UNSURE_DECISIVE_CHANCE) * (1 + DECISIVE_BONUS)
        out.append(Signal(CONFIRM, f, (value,), detail="; ".join(reasons), priority=score(f, chance)))

    return sorted(out, key=lambda s: -s.priority)
