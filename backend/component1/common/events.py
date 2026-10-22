"""Event sourcing for the Living Blueprint.

Every change is stored as an Event. The current blueprint is the result of
replaying all events in order, so no old value is ever lost.

This is an in-memory version that mirrors the blueprint_events table in
db/schema.sql. A PostgreSQL version can replace it later without changing
the rest of the code.
"""
from __future__ import annotations

from typing import Optional

from .schema import (
    Blueprint,
    Decision,
    DecisionStatus,
    Event,
    EventType,
    FieldName,
    FieldState,
    FieldStatus,
    Provenance,
)

# Starter rules for decision status changes. The formal state machine
# (feature 11) will refine this later.
ALLOWED_TRANSITIONS: dict[DecisionStatus, set[DecisionStatus]] = {
    DecisionStatus.PROPOSED: {
        DecisionStatus.CONFIRMED,
        DecisionStatus.FINALIZED,
        DecisionStatus.REJECTED,
        DecisionStatus.SUPERSEDED,
    },
    DecisionStatus.CONFIRMED: {
        DecisionStatus.FINALIZED,
        DecisionStatus.REJECTED,
        DecisionStatus.SUPERSEDED,
        DecisionStatus.PENDING_FIX,      # Domino validation: an earlier choice it depends on changed
    },
    DecisionStatus.FINALIZED: {DecisionStatus.SUPERSEDED, DecisionStatus.PENDING_FIX},
    # After a re-check the decision goes back to what it was, is re-confirmed, or is replaced.
    DecisionStatus.PENDING_FIX: {DecisionStatus.CONFIRMED, DecisionStatus.FINALIZED, DecisionStatus.SUPERSEDED,
                                 DecisionStatus.REJECTED},
    DecisionStatus.REJECTED: set(),
    DecisionStatus.SUPERSEDED: set(),
}


def apply_event(state: Blueprint, event: Event) -> Blueprint:
    """Return a NEW blueprint with the event applied. Does not change `state`."""
    if event.version != state.version + 1:
        raise ValueError(
            f"event version {event.version} does not follow blueprint version {state.version}"
        )
    new = state.model_copy(deep=True)
    p = event.payload

    if event.event_type == EventType.FIELD_UPDATED:
        name = FieldName(p["field"])
        old = new.fields[name]
        # Build a new FieldState so its validators run (value/status, confidence range).
        new.fields[name] = FieldState(
            name=name,
            value=p["new_value"],
            status=FieldStatus(p["new_status"]),
            confidence=p["confidence"],
            provenance=old.provenance + ([Provenance.model_validate(p["provenance"])] if p.get("provenance") else []),
            updated_at=event.created_at,
        )

    elif event.event_type == EventType.DECISION_RECORDED:
        d = Decision.model_validate(p["decision"])
        new.decisions[d.decision_id] = d

    elif event.event_type in (EventType.DECISION_STATUS_CHANGED, EventType.DECISION_UPDATED):
        d = new.decisions[p["decision_id"]]
        if event.event_type == EventType.DECISION_STATUS_CHANGED:
            d.status = DecisionStatus(p["new_status"])
        if "agreed_by" in p:
            d.agreed_by = list(p["agreed_by"])
        if "objected_by" in p:
            d.objected_by = list(p["objected_by"])
        if "rationale" in p:
            d.rationale = p["rationale"]
        if "pending_reason" in p:
            d.pending_reason = p["pending_reason"]
        if "held_status" in p:
            d.held_status = DecisionStatus(p["held_status"]) if p["held_status"] else None
        d.updated_at = event.created_at

    elif event.event_type == EventType.SCORES_UPDATED:
        if "novelty_score" in p:
            new.novelty_score = p["novelty_score"]
        if "feasibility" in p:
            new.feasibility = p["feasibility"]

    new.version = event.version
    new.updated_at = event.created_at
    return new


def replay(project_id: str, events: list[Event]) -> Blueprint:
    """Rebuild a blueprint from scratch by applying events in order."""
    state = Blueprint(project_id=project_id)
    for event in sorted(events, key=lambda e: e.version):
        state = apply_event(state, event)
    return state


class EventStore:
    """Holds the event log and the current blueprint for one project."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self._events: list[Event] = []
        self._state = Blueprint(project_id=project_id)

    @classmethod
    def from_events(cls, project_id: str, events: list[Event]) -> "EventStore":
        """Rebuild a store from events that were saved earlier (for example after a restart)."""
        store = cls(project_id)
        ordered = sorted(events, key=lambda e: e.version)
        store._state = replay(project_id, ordered)
        store._events = ordered
        return store

    # -- read access (copies, so callers cannot change history by accident)
    @property
    def state(self) -> Blueprint:
        return self._state.model_copy(deep=True)

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    # -- writing
    def _append(self, event_type: EventType, payload: dict, actor: str) -> Event:
        event = Event(
            project_id=self.project_id,
            version=self._state.version + 1,
            event_type=event_type,
            payload=payload,
            actor=actor,
        )
        self._state = apply_event(self._state, event)  # raises before we store if invalid
        self._events.append(event)
        return event

    def update_field(
        self,
        field_name: FieldName,
        value: str,
        status: FieldStatus,
        confidence: float,
        provenance: Optional[Provenance],
        actor: str,
    ) -> Event:
        if status == FieldStatus.MISSING:
            raise ValueError("use a status other than 'missing' when setting a value")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        current = self._state.fields[field_name]
        payload = {
            "field": field_name.value,
            "old_value": current.value,
            "new_value": value,
            "old_status": current.status.value,
            "new_status": status.value,
            "confidence": confidence,
            "provenance": provenance.model_dump(mode="json") if provenance is not None else None,
        }
        return self._append(EventType.FIELD_UPDATED, payload, actor)

    def record_decision(self, decision: Decision, actor: str) -> Event:
        if decision.decision_id in self._state.decisions:
            raise ValueError(f"decision {decision.decision_id} already exists")
        return self._append(
            EventType.DECISION_RECORDED,
            {"decision": decision.model_dump(mode="json")},
            actor,
        )

    def change_decision_status(
        self,
        decision_id: str,
        new_status: DecisionStatus,
        actor: str,
        agreed_by: Optional[list[str]] = None,
        **extra,
    ) -> Event:
        """extra may carry objected_by, rationale, pending_reason and held_status, changed together with the status."""
        if decision_id not in self._state.decisions:
            raise KeyError(f"unknown decision {decision_id}")
        extra = {k: (v.value if hasattr(v, "value") else v) for k, v in extra.items()}
        bad = set(extra) - {"objected_by", "rationale", "pending_reason", "held_status"}
        if bad:
            raise ValueError(f"cannot change {sorted(bad)} with a status change")
        old_status = self._state.decisions[decision_id].status
        if new_status not in ALLOWED_TRANSITIONS[old_status]:
            raise ValueError(f"cannot move a decision from {old_status.value} to {new_status.value}")
        payload: dict = {
            "decision_id": decision_id,
            "old_status": old_status.value,
            "new_status": new_status.value,
        }
        if agreed_by is not None:
            payload["agreed_by"] = agreed_by
        payload.update(extra)
        return self._append(EventType.DECISION_STATUS_CHANGED, payload, actor)

    def update_decision(self, decision_id: str, actor: str, **changes) -> Event:
        """Change who agreed / objected, the reason, or the re-check note, WITHOUT changing the status.
        changes: agreed_by, objected_by, rationale, pending_reason, held_status."""
        if decision_id not in self._state.decisions:
            raise KeyError(f"unknown decision {decision_id}")
        allowed = {"agreed_by", "objected_by", "rationale", "pending_reason", "held_status"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"cannot update {sorted(unknown)}")
        if not changes:
            raise ValueError("nothing to update")
        payload = {"decision_id": decision_id, **{k: (v.value if hasattr(v, "value") else v) for k, v in changes.items()}}
        return self._append(EventType.DECISION_UPDATED, payload, actor)

    def update_scores(
        self,
        actor: str,
        novelty_score: Optional[float] = None,
        feasibility: Optional[dict[str, str]] = None,
    ) -> Event:
        payload: dict = {}
        if novelty_score is not None:
            payload["novelty_score"] = novelty_score
        if feasibility is not None:
            payload["feasibility"] = feasibility
        if not payload:
            raise ValueError("nothing to update")
        return self._append(EventType.SCORES_UPDATED, payload, actor)

    # -- checks
    def rebuild(self) -> Blueprint:
        """Replay every event from scratch. Should always equal the current state."""
        return replay(self.project_id, self._events)