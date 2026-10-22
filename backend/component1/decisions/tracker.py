"""Create decisions and move them through their life, from what people say.

    changes = on_statement(store, "algorithm", "XGBoost", "M-1", "m7", text, intent="proposal")
    changes = on_reply(store, decision, "M-2", "agreement", "m8")

Rules (they follow the labeling guide's decision intent labels):
  * Stating a value no decision has yet creates a PROPOSED decision, with its reason if the message
    gives one ("... because it handles tabular data well").
  * Someone ELSE stating the same value, or agreeing, adds them to agreed_by and moves
    proposed -> confirmed (and pending_fix -> confirmed: the re-check is done).
  * Finalizing ("ok, lock it in") moves it to finalized, but only once someone other than the
    proposer is on board: one person cannot finalize their own idea alone.
  * Someone else rejecting it adds them to objected_by; a proposal nobody agreed with is rejected.
  * When a value is confirmed or finalized, the other open proposals for that field are superseded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..common.events import EventStore
from ..common.schema import Decision, DecisionStatus, FieldName
from ..conflicts.same import same_value

ACTOR = "decision_tracker"
ACTIVE = (DecisionStatus.PROPOSED, DecisionStatus.CONFIRMED, DecisionStatus.FINALIZED, DecisionStatus.PENDING_FIX)
SETTLED = (DecisionStatus.CONFIRMED, DecisionStatus.FINALIZED)

# "... because it handles tabular data well", "since we only have 3 weeks", "so that it is explainable"
_REASON = re.compile(
    r"\b(?:because(?:\s+of)?|since|so\s+that|due\s+to|given\s+that|as\s+(?:it|they|we|this|the\s+data)|"
    r"'?cause|coz|bc)\b[\s,:]*(?P<why>[^.!?;]{3,200})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Change:
    decision_id: str
    field: str
    value: str
    kind: str          # proposed / agreed / confirmed / finalized / objected / rejected / superseded / reconfirmed
    by: str


def reason_in(text: str) -> Optional[str]:
    """The reason a message gives, or None."""
    m = _REASON.search(text or "")
    if not m:
        return None
    why = " ".join(m.group("why").split()).strip(" ,-")
    return why or None


def is_planning_decision(d: Decision) -> bool:
    return not d.decision_id.startswith("ms:")       # milestones are decisions too, handled elsewhere


def decisions_for(state, field: str) -> list[Decision]:
    return sorted((d for d in state.decisions.values() if is_planning_decision(d) and d.field.value == field),
                  key=lambda d: d.created_at)


def active_for(state, field: str) -> list[Decision]:
    return [d for d in decisions_for(state, field) if d.status in ACTIVE]


def find_active(state, field: str, value: str) -> Optional[Decision]:
    for d in reversed(active_for(state, field)):
        if same_value(field, d.value, value):
            return d
    return None


def current_decision(state, field: str) -> Optional[Decision]:
    """The decision the field's plan rests on: finalized first, then confirmed, re-check, proposed."""
    rank = {DecisionStatus.FINALIZED: 0, DecisionStatus.CONFIRMED: 1, DecisionStatus.PENDING_FIX: 2,
            DecisionStatus.PROPOSED: 3}
    items = active_for(state, field)
    return min(items, key=lambda d: (rank[d.status], -d.updated_at.timestamp())) if items else None


def _decision(store: EventStore, decision_id: str) -> Decision:
    return store.state.decisions[decision_id]


def _supersede_others(store: EventStore, kept: Decision, by: str) -> list[Change]:
    out = []
    for d in active_for(store.state, kept.field.value):
        if d.decision_id != kept.decision_id:
            store.change_decision_status(d.decision_id, DecisionStatus.SUPERSEDED, ACTOR,
                                         pending_reason=None, held_status=None)
            out.append(Change(d.decision_id, d.field.value, d.value, "superseded", by))
    return out


def propose(store: EventStore, field: str, value: str, speaker: str, source_id: str,
            rationale: Optional[str] = None, status: DecisionStatus = DecisionStatus.PROPOSED,
            agreed_by: Optional[list[str]] = None) -> Change:
    decision = Decision(
        decision_id=f"d:{field}:{store.state.version + 1}",
        field=FieldName(field),
        value=value,
        status=status,
        proposed_by=speaker,
        agreed_by=list(agreed_by or []),
        rationale=rationale,
        source_id=source_id,
    )
    store.record_decision(decision, ACTOR)
    return Change(decision.decision_id, field, value, "proposed" if status == DecisionStatus.PROPOSED else status.value,
                  speaker)


def endorse(store: EventStore, d: Decision, speaker: str, finalize: bool = False) -> list[Change]:
    """Someone other than the proposer backs a decision (agreement or finalization)."""
    if speaker == d.proposed_by:
        if finalize and d.agreed_by and d.status in (DecisionStatus.PROPOSED, DecisionStatus.CONFIRMED,
                                                     DecisionStatus.PENDING_FIX):
            store.change_decision_status(d.decision_id, DecisionStatus.FINALIZED, ACTOR,
                                         pending_reason=None, held_status=None)
            return [Change(d.decision_id, d.field.value, d.value, "finalized", speaker),
                    *_supersede_others(store, _decision(store, d.decision_id), speaker)]
        return []                                          # agreeing with yourself changes nothing
    agreed = d.agreed_by + ([speaker] if speaker not in d.agreed_by else [])
    objected = [o for o in d.objected_by if o != speaker]   # changing your mind removes the objection
    changes: list[Change] = []
    target = DecisionStatus.FINALIZED if finalize else (
        DecisionStatus.CONFIRMED if d.status in (DecisionStatus.PROPOSED, DecisionStatus.PENDING_FIX) else None)
    if target is not None and target != d.status:
        store.change_decision_status(d.decision_id, target, ACTOR, agreed_by=agreed, objected_by=objected,
                                     pending_reason=None, held_status=None)
        kind = "reconfirmed" if d.status == DecisionStatus.PENDING_FIX and target == DecisionStatus.CONFIRMED \
            else target.value
        changes.append(Change(d.decision_id, d.field.value, d.value, kind, speaker))
        changes += _supersede_others(store, _decision(store, d.decision_id), speaker)
    elif speaker not in d.agreed_by or objected != d.objected_by:
        store.update_decision(d.decision_id, ACTOR, agreed_by=agreed, objected_by=objected)
        changes.append(Change(d.decision_id, d.field.value, d.value, "agreed", speaker))
    return changes


def object_to(store: EventStore, d: Decision, speaker: str) -> list[Change]:
    """Someone other than the proposer rejects a decision."""
    if speaker == d.proposed_by or speaker in d.objected_by:
        return []
    objected = d.objected_by + [speaker]
    agreed = [a for a in d.agreed_by if a != speaker]
    if d.status == DecisionStatus.PROPOSED and not agreed:
        store.change_decision_status(d.decision_id, DecisionStatus.REJECTED, ACTOR, agreed_by=agreed,
                                     objected_by=objected)
        return [Change(d.decision_id, d.field.value, d.value, "rejected", speaker)]
    store.update_decision(d.decision_id, ACTOR, agreed_by=agreed, objected_by=objected)
    return [Change(d.decision_id, d.field.value, d.value, "objected", speaker)]


def on_statement(store: EventStore, field: str, value: str, speaker: str, source_id: str, text: str,
                 intent: Optional[str] = None) -> list[Change]:
    """A message states a value for a field."""
    if field == FieldName.TIMELINE.value and value.lower().startswith("milestone: "):
        return []
    existing = find_active(store.state, field, value)
    why = reason_in(text)
    if existing is None:
        return [propose(store, field, value, speaker, source_id, why)]
    if intent == "rejection":              # "No, AWS costs too much": naming a value to turn it down
        return object_to(store, existing, speaker)
    changes = endorse(store, existing, speaker, finalize=intent == "finalization")
    current = _decision(store, existing.decision_id)
    if why and not current.rationale:
        store.update_decision(current.decision_id, ACTOR, rationale=why)
    return changes


def on_reply(store: EventStore, d: Decision, speaker: str, intent: str) -> list[Change]:
    """A message with no value of its own answers a decision ("agreed", "lock it in", "no")."""
    if d.status not in ACTIVE:
        return []
    if intent in ("agreement", "finalization"):
        return endorse(store, d, speaker, finalize=intent == "finalization")
    if intent == "rejection":
        return object_to(store, d, speaker)
    return []


def record_document_decision(store: EventStore, field: str, value: str, filename: str, page: Optional[int],
                             span: str, accepted_by: str, source_id: str) -> list[Change]:
    """An accepted document suggestion: confirmed, proposed by the document, agreed by the member."""
    where = f"{filename} p.{page}" if page else filename
    existing = find_active(store.state, field, value)
    if existing is not None:
        return endorse(store, existing, accepted_by)
    change = propose(store, field, value, f"doc:{filename}", source_id,
                     rationale=f'Found in {where}: "{span}"', status=DecisionStatus.CONFIRMED, agreed_by=[accepted_by])
    return [change, *_supersede_others(store, _decision(store, change.decision_id), accepted_by)]


def turned_down(state, field: str, value: Optional[str]) -> bool:
    """The value has decision records and none of them is still active (all rejected or replaced)."""
    mine = [d for d in decisions_for(state, field) if same_value(field, d.value, value)]
    return bool(mine) and all(d.status not in ACTIVE for d in mine)


def resolved_by_decision(state, field: str, label: str) -> str:
    """The state classifier counts every mention ever made, so after a disagreement it keeps saying
    "conflicting" even once the team settled it, and a value agreed with a plain "Agreed." (not repeated)
    stays "known". If the field has a settled decision and every rival value was replaced or rejected,
    the team has decided: the field is confirmed."""
    value = state.fields[FieldName(field)].value
    if label in ("confirmed", "known") and value and find_active(state, field, value) is None and any(
            d.status == DecisionStatus.REJECTED and same_value(field, d.value, value) for d in decisions_for(state, field)):
        return "uncertain"                 # the value on the field was turned down, so it is not settled
    if label == "conflicting" and value and any(d.status not in ACTIVE for d in decisions_for(state, field))             and not any(not same_value(field, d.value, value) for d in active_for(state, field)):
        label = "known"                    # every rival value was rejected or replaced: no one still holds one
    if label not in ("conflicting", "known", "uncertain"):
        return label
    current = current_decision(state, field)
    if current is None or current.status not in SETTLED:
        return label
    # A new value said after the decision ("what about a neural network?") is a new disagreement,
    # even before it is recorded as a proposal: the field is judged before the decision records are.
    if value and not same_value(field, value, current.value):
        return label
    rivals = [d for d in active_for(state, field) if d.decision_id != current.decision_id]
    return "confirmed" if not rivals else label


def effective_state(state):
    """A copy of the blueprint where each field with a settled decision takes that decision's value
    and counts as confirmed. Domino validation checks the plan the team actually agreed on."""
    from ..common.schema import FieldStatus
    copy = state.model_copy(deep=True)
    for name in FieldName:
        d = current_decision(state, name.value)
        if d is not None and d.status in SETTLED:
            f = copy.fields[name]
            copy.fields[name] = f.model_copy(update={"value": d.value, "status": FieldStatus.CONFIRMED})
    return copy
