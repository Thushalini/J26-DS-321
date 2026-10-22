"""Decisions as the Blueprint tab shows them, each with its own timeline built from the event log."""
from __future__ import annotations

from typing import Iterable, Optional

from ..common.schema import DecisionStatus, FieldName
from .tracker import current_decision, is_planning_decision

STATUS_WORDS = {
    "proposed": "proposed", "confirmed": "confirmed", "finalized": "finalized", "rejected": "rejected",
    "superseded": "replaced", "pending_fix": "needs re-check",
}


def _plain(x):
    return getattr(x, "value", x)


def decision_timelines(events: Iterable) -> dict[str, list[dict]]:
    """decision_id -> what happened to it, oldest first."""
    out: dict[str, list[dict]] = {}
    for e in sorted(events, key=lambda e: e.version):
        kind = _plain(e.event_type)
        p = e.payload
        if kind == "decision_recorded":
            d = p["decision"]
            out.setdefault(d["decision_id"], []).append(
                {"time": e.created_at.isoformat(), "what": d.get("status", "proposed"), "by": d.get("proposed_by")})
        elif kind == "decision_status_changed":
            out.setdefault(p["decision_id"], []).append(
                {"time": e.created_at.isoformat(), "what": p["new_status"], "from": p.get("old_status"),
                 "reason": p.get("pending_reason")})
        elif kind == "decision_updated":
            item = {"time": e.created_at.isoformat(), "what": "updated"}
            for key in ("agreed_by", "objected_by", "rationale"):
                if key in p:
                    item[key] = p[key]
            out.setdefault(p["decision_id"], []).append(item)
    return out


def decision_dict(d, timeline: Optional[list[dict]] = None, include_evidence: bool = True) -> dict:
    status = d.status.value
    out = {
        "decision_id": d.decision_id,
        "field": d.field.value,
        "label": d.field.value.replace("_", " ").capitalize(),
        "value": d.value,
        "status": status,
        "status_text": STATUS_WORDS.get(status, status),
        "proposed_by": d.proposed_by,
        "agreed_by": list(d.agreed_by),
        "objected_by": list(d.objected_by),
        "rationale": d.rationale if include_evidence else None,
        "supersedes": d.supersedes,
        "pending_reason": d.pending_reason,
        "created_at": d.created_at.isoformat(),
        "updated_at": d.updated_at.isoformat(),
    }
    if timeline is not None:
        out["timeline"] = timeline
    return out


def decisions_view(state, events: Iterable, include_evidence: bool = True, hide_documents: bool = False) -> dict:
    """{"current": one decision per field the team is working on, "all": every decision, newest first}."""
    events = list(events)
    timelines = decision_timelines(events)
    items = [d for d in state.decisions.values() if is_planning_decision(d)]

    def show(d):
        out = decision_dict(d, timelines.get(d.decision_id, []), include_evidence)
        if hide_documents and str(out["proposed_by"]).startswith("doc:"):
            out["proposed_by"] = "a document"
            out["rationale"] = None
        return out

    current = []
    for name in FieldName:
        d = current_decision(state, name.value)
        if d is not None:
            current.append(show(d))
    everything = [show(d) for d in sorted(items, key=lambda d: d.updated_at, reverse=True)]
    return {
        "current": current,
        "all": everything,
        "needs_recheck": [x for x in current if x["status"] == DecisionStatus.PENDING_FIX.value],
    }
