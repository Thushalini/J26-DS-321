"""The timeline field as it stands now, worked out into dates (used by the plan and the plan checks)."""
from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

from ..common.schema import FieldName
from ..state.normalize import normalize_value
from .dates import resolve


def first_said(events: Iterable, value: Optional[str]):
    """When the timeline's current value was first said. "8 weeks" counts from that moment."""
    if not value:
        return None
    wanted = normalize_value(value)
    for e in sorted(events, key=lambda e: e.version):
        p = e.payload
        if (getattr(e.event_type, "value", e.event_type) == "field_updated" and e.actor == "extractor"
                and p.get("field") == FieldName.TIMELINE.value and normalize_value(p.get("new_value") or "") == wanted):
            return e.created_at
    return None


def current_timeline(state, events: Iterable, today: Optional[date] = None) -> Optional[dict]:
    """resolve() of the timeline field's current value, or None if there is none."""
    events = list(events)
    value = state.fields[FieldName.TIMELINE].value
    return resolve(value, said_on=first_said(events, value), today=today)
