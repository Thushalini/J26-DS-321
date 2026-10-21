"""The planning blueprint as the Blueprint tab needs it: all nine fields, how settled each one
is, and what is still missing. Built from the saved events (c1_events), so it always matches
what Buddy said in the chat.

Nothing here writes anything. The chat text behind a value ("evidence") is only included when
the caller says the viewer may see the team's chat.
"""
from __future__ import annotations

from typing import Any

from .common.events import EventStore
from .common.schema import FieldName, FieldStatus
from .gaps.gaps import check_completeness, ready_gaps
from .state.features import mentions_for_field
from .storage import load_events

NOTE = "Confidence is the system's own estimate (High, Medium or Low). It is not a guarantee of being right."


def label(name: FieldName) -> str:
    return name.value.replace("_", " ").capitalize()


def confidence_band(confidence: float) -> str:
    """Display words only. The number behind them is not calibrated."""
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.6:
        return "Medium"
    return "Low"


def plan_from_events(team_id: str, events: list, include_evidence: bool = False) -> dict[str, Any]:
    store = EventStore.from_events(team_id, events)
    state = store.state
    report = check_completeness(state)

    fields = []
    for name in FieldName:
        f = state.fields[name]
        item: dict[str, Any] = {
            "name": name.value,
            "label": label(name),
            "status": f.status.value,
            "value": f.value,
            "confidence_band": confidence_band(f.confidence) if f.status != FieldStatus.MISSING else None,
            "mentions": len(f.provenance),
        }
        if f.status == FieldStatus.CONFLICTING:
            seen: dict[str, str] = {}
            for m in mentions_for_field(store.events, name):
                seen.setdefault(m["value"], m["original"])
            item["candidates"] = list(seen.values())
        if include_evidence:
            item["evidence"] = [
                {
                    "speaker": p.speaker,
                    "text": p.text_span,
                    "source_id": p.source_id,
                    "source_type": p.source_type.value,
                    "time": p.timestamp.isoformat() if p.timestamp else None,
                }
                for p in f.provenance
            ]
        fields.append(item)

    confirmed = sum(1 for f in state.fields.values() if f.status == FieldStatus.CONFIRMED)
    return {
        "team_id": team_id,
        "version": state.version,
        "updated_at": state.updated_at.isoformat() if state.updated_at else None,
        "readiness": {
            "complete": report.complete,
            "confirmed": confirmed,
            "total": len(FieldName),
            "missing": [n.value for n in report.missing],
            "conflicting": [n.value for n in report.conflicting],
            "not_confirmed": [n.value for n in report.not_confirmed],
            "low_confidence": [n.value for n in report.low_confidence],
        },
        "next_gaps": [n.value for n in ready_gaps(state)[:3]],
        "fields": fields,
        "evidence_included": include_evidence,
        "note": NOTE,
    }


def build_plan(conn: Any, team_id: str, include_evidence: bool = False) -> dict[str, Any]:
    """The plan for one team, read from the database. A team Buddy has not heard yet gets an empty plan."""
    return plan_from_events(team_id, load_events(conn, team_id), include_evidence)