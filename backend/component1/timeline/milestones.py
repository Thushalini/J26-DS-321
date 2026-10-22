"""Named milestones (demo, presentation, mid evaluation, ...) as decisions on the timeline field.

    record_milestone(store, "milestone: Demo, 2026-11-20", span="demo is on 20/11", speaker="M-1", source_id="m7")

One decision per milestone name. If the same milestone is said again with the same date, nothing
changes. With a NEW date, a new decision is recorded that supersedes the old one, so the history
keeps both ("Demo moved from 20 Nov to 27 Nov").
"""
from __future__ import annotations

import re
from typing import Optional

from ..common.events import EventStore
from ..common.schema import Decision, DecisionStatus, Event, FieldName
from ..state.normalize import normalize_value
from .dates import MILESTONE_PREFIX, resolve_milestone

INACTIVE = (DecisionStatus.SUPERSEDED, DecisionStatus.REJECTED)
ACTOR = "extractor"


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "milestone"


def active_milestones(store_or_state) -> list[Decision]:
    state = store_or_state.state if isinstance(store_or_state, EventStore) else store_or_state
    return [d for d in state.decisions.values()
            if d.field == FieldName.TIMELINE and d.decision_id.startswith("ms:") and d.status not in INACTIVE]


def record_milestone(store: EventStore, value: str, span: str, speaker: str, source_id: str) -> Optional[Event]:
    """Record a milestone said in the chat. Returns the new decision's event, or None if nothing changed."""
    parsed = resolve_milestone(value)
    if parsed is None or not parsed["when"]:
        return None
    body = value[len(MILESTONE_PREFIX):].strip()
    key = _key(parsed["name"])
    same_name = [d for d in active_milestones(store) if d.decision_id.startswith(f"ms:{key}:")]
    if same_name and normalize_value(same_name[-1].value) == normalize_value(body):
        return None
    decision = Decision(
        decision_id=f"ms:{key}:{store.state.version + 1}",
        field=FieldName.TIMELINE,
        value=body,
        proposed_by=speaker,
        rationale=f"{span} ({source_id})",
        supersedes=same_name[-1].decision_id if same_name else None,
    )
    event = store.record_decision(decision, ACTOR)
    for old in same_name:
        store.change_decision_status(old.decision_id, DecisionStatus.SUPERSEDED, ACTOR)
    return event


def milestone_list(state) -> list[dict]:
    """The active milestones, soonest first (milestones with only a week number or no date last)."""
    out = []
    for d in active_milestones(state):
        parsed = resolve_milestone(MILESTONE_PREFIX + d.value, said_on=d.created_at) or {}
        out.append({
            "decision_id": d.decision_id,
            "name": parsed.get("name") or d.value,
            "when": parsed.get("when"),
            "date": parsed.get("date"),
            "week": parsed.get("week"),
            "proposed_by": d.proposed_by,
            "status": d.status.value,
            "said_at": d.created_at.isoformat(),
        })
    out.sort(key=lambda m: (m["date"] is None, m["date"] or "", m["week"] if m["week"] is not None else 99))
    return out
