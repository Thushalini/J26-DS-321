"""The blueprint's decision history, readable: who said what, when, and how each field's status moved.

    entries = field_history(events)                        # every change, oldest first
    entries = field_history(events, field="algorithm")     # one field only

Built only from the saved event log (c1_events), so it always matches what the blueprint shows.
One chat message usually writes two events: the extractor's mention, then the state classifier's
judgement. They become ONE entry ("M-2 proposed XGBoost; status: missing -> known"). An accepted
document suggestion writes three (document, acceptance, judgement) and also becomes one entry.

Entry kinds:
    proposed          the first value for a field
    changed           a different value from the one before ("from XGBoost to LSTM")
    agreed            the same value again, from someone else
    repeated          the same value again, from the same person (this is NOT confirmation)
    accepted_document a document suggestion accepted by a team member
    status_changed    only the status moved, with no new mention
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .common.schema import FieldName
from .documents.integrate import DOCUMENT_SPEAKER_PREFIX
from .pipeline import STATE_ACTOR
from .state.features import EXTRACTOR_ACTOR
from .state.normalize import normalize_value

ACCEPTED_PREFIX = "Accepted from "


def _plain(x):
    return getattr(x, "value", x)


def _label(field: str) -> str:
    return field.replace("_", " ").capitalize()


def _band(confidence: Optional[float]) -> Optional[str]:
    if confidence is None:
        return None
    return "High" if confidence >= 0.85 else "Medium" if confidence >= 0.6 else "Low"


def _summary(entry: dict) -> str:
    who, value, before = entry["by"] or "Someone", entry["value"], entry["previous_value"]
    kind = entry["kind"]
    if kind == "proposed":
        return f"{who} proposed {value}"
    if kind == "changed":
        return f"{who} changed it from {before} to {value}"
    if kind == "agreed":
        return f"{who} agreed on {value}"
    if kind == "repeated":
        return f"{who} repeated {value}"
    if kind == "accepted_document":
        return f"{entry['accepted_by'] or 'A team member'} accepted {value} from {entry['source_label']}"
    return f"Status changed to {entry['status_after']}"


def _source(prov: dict) -> dict:
    return {"type": prov.get("source_type"), "id": prov.get("source_id"), "page": prov.get("page")}


def field_history(events: Iterable[Any], field: Optional[str] = None, include_evidence: bool = False,
                  hide_documents: bool = False) -> list[dict]:
    """Every change, oldest first. include_evidence adds the exact words; hide_documents removes file
    names (for viewers who must not see anything that could identify a person)."""
    state: dict[str, dict] = {n.value: {"value": None, "status": "missing", "speaker": None} for n in FieldName}
    entries: list[dict] = []
    open_entry: dict[str, dict] = {}            # field -> the entry the next judgement belongs to

    def close(f: str) -> None:
        e = open_entry.pop(f, None)
        if e is not None:
            e["summary"] = _summary(e)
            entries.append(e)

    for event in sorted(events, key=lambda e: e.version):
        if _plain(event.event_type) != "field_updated":
            continue
        p = event.payload
        f = p.get("field")
        if f not in state:
            continue
        prov = p.get("provenance") or {}
        now = state[f]

        if event.actor == STATE_ACTOR:          # the judgement after a mention
            e = open_entry.get(f)
            if e is None:
                if p.get("new_status") == now["status"]:
                    continue
                e = _new_entry(event, f, "status_changed", now, by=None, prov={})
                open_entry[f] = e
            e["status_after"] = p.get("new_status")
            e["confidence_band"] = _band(p.get("confidence"))
            now["status"] = p.get("new_status")
            close(f)
            continue
        if event.actor != EXTRACTOR_ACTOR:
            continue

        speaker = prov.get("speaker")
        span = prov.get("text_span") or ""
        pending = open_entry.get(f)
        # The second half of an accepted document suggestion: the member's acceptance.
        if pending is not None and pending["kind"] == "accepted_document" and span.startswith(ACCEPTED_PREFIX):
            pending["accepted_by"] = speaker
            now["status"] = p.get("new_status", now["status"])
            continue
        close(f)

        new_value = p.get("new_value")
        if speaker and str(speaker).startswith(DOCUMENT_SPEAKER_PREFIX):
            kind = "accepted_document"
        elif now["value"] is None:
            kind = "proposed"
        elif normalize_value(new_value or "") != normalize_value(now["value"] or ""):
            kind = "changed"
        elif speaker == now["speaker"]:
            kind = "repeated"
        else:
            kind = "agreed"
        e = _new_entry(event, f, kind, now, by=speaker, prov=prov)
        e["value"] = new_value
        e["status_after"] = p.get("new_status", now["status"])     # the judgement, if one follows, replaces it
        if kind == "accepted_document":
            filename = str(speaker)[len(DOCUMENT_SPEAKER_PREFIX):]
            e["by"] = "document"
            e["document"] = filename
            e["source_label"] = f"{filename} p.{prov['page']}" if prov.get("page") else filename
        if include_evidence:
            e["text"] = span
        open_entry[f] = e
        now["value"], now["speaker"] = new_value, speaker
        now["status"] = p.get("new_status", now["status"])

    for f in list(open_entry):                  # a mention with no judgement after it
        close(f)

    if hide_documents:
        for e in entries:
            if e["kind"] == "accepted_document":
                e["document"], e["source_label"] = None, "a document"
                e["summary"] = _summary(e)
    entries.sort(key=lambda e: e["version"])
    return [e for e in entries if field is None or e["field"] == field]


def _new_entry(event: Any, field: str, kind: str, now: dict, by: Optional[str], prov: dict) -> dict:
    return {
        "version": event.version,
        "time": event.created_at.isoformat(),
        "field": field,
        "label": _label(field),
        "kind": kind,
        "value": now["value"],
        "previous_value": now["value"],
        "status_before": now["status"],
        "status_after": now["status"],
        "confidence_band": None,
        "by": by,
        "accepted_by": None,
        "source": _source(prov) if prov else None,
        "source_label": None,
        "document": None,
    }


def recent_changes(events: Iterable[Any], limit: int = 10, include_evidence: bool = False,
                   hide_documents: bool = False) -> list[dict]:
    """The latest changes across all fields, newest first."""
    entries = field_history(events, include_evidence=include_evidence, hide_documents=hide_documents)
    return list(reversed(entries))[:limit]
