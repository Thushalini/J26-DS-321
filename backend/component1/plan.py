"""The planning blueprint as the Blueprint tab needs it: all ten fields, how settled each one
is, and what is still missing. Built from the saved events (c1_events), so it always matches
what Buddy said in the chat.

Nothing here writes anything. The chat text behind a value ("evidence") is only included when
the caller says the viewer may see the team's chat.

Each field also lists its open document suggestions ("suggestions", best first), and the plan
lists the team's uploaded documents. Both are left out when include_documents is False.
"""
from __future__ import annotations

from typing import Any, Optional

from .common.events import EventStore
from .common.schema import FieldName, FieldStatus
from .documents.candidates import Candidate, group_candidates
from .gaps.gaps import check_completeness, ready_gaps
from .checks.plan_checks import run_checks
from .history import recent_changes
from .timeline.current import first_said
from .timeline.dates import resolve
from .timeline.milestones import milestone_list
from .state.features import mentions_for_field
from .storage import list_doc_facts, list_documents, load_events

MAX_SUGGESTIONS_PER_FIELD = 3
RECENT_CHANGES = 10

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


def plan_from_events(
    team_id: str,
    events: list,
    include_evidence: bool = False,
    suggestions: Optional[dict[str, list[Candidate]]] = None,
    documents: Optional[list[dict]] = None,
    hide_documents: bool = False,
) -> dict[str, Any]:
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
        if name == FieldName.TIMELINE:
            item["timeline"] = resolve(f.value, said_on=first_said(store.events, f.value))
        if suggestions is not None:
            item["suggestions"] = [c.as_dict(include_evidence)
                                   for c in suggestions.get(name.value, [])[:MAX_SUGGESTIONS_PER_FIELD]]
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
        "recent_changes": recent_changes(events, RECENT_CHANGES, include_evidence, hide_documents),
        "milestones": milestone_list(state),
        "checks": [r.as_dict() for r in run_checks(state, store.events)],
        "feasibility": state.feasibility,
        "documents": documents,
        "note": NOTE,
    }


DOCUMENT_KEYS = ("doc_id", "filename", "status", "error", "pages", "facts", "uploaded_at", "processed_at")


def build_plan(conn: Any, team_id: str, include_evidence: bool = False,
               include_documents: bool = True) -> dict[str, Any]:
    """The plan for one team, read from the database. A team Buddy has not heard yet gets an empty plan."""
    suggestions = documents = None
    if include_documents:
        suggestions = group_candidates(list_doc_facts(conn, team_id))
        documents = [{k: d.get(k) for k in DOCUMENT_KEYS} for d in list_documents(conn, team_id)]
    return plan_from_events(team_id, load_events(conn, team_id), include_evidence, suggestions, documents,
                            hide_documents=not include_documents)

def blueprint_for_component4(conn: Any, team_id: str) -> dict[str, Any]:
    """The blueprint in the shape Component 4's c1_adapter reads (fields, version, novelty_score,
    feasibility), rebuilt from the saved events. For example, in Component 4:

        from component1.plan import blueprint_for_component4
        project = assemble(blueprint=blueprint_for_component4(conn, team_id), ...)
    """
    return EventStore.from_events(team_id, load_events(conn, team_id)).state.model_dump(mode="json")
