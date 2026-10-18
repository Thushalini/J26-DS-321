"""Gap detection (feature 5) and the completeness check (feature 17).

Both are plain rules. They read the blueprint and never change it.
"""
from __future__ import annotations

from pydantic import BaseModel

from ..common.schema import (
    BASE_WEIGHT,
    DEPENDS_ON,
    Blueprint,
    FieldName,
    FieldStatus,
)

# A field counts as "reliable enough to build on" once it is known or confirmed.
RELIABLE = {FieldStatus.KNOWN, FieldStatus.CONFIRMED}

# Starting values, set by hand. Explain and justify them in the report.
REQUIRED_FIELDS = list(FieldName)
MIN_CONFIDENCE = 0.7


def ready_gaps(state: Blueprint) -> list[FieldName]:
    """Missing fields that can sensibly be asked about now, most important first.

    A missing field is ready when every field it depends on is known or confirmed.
    Ties keep the order the fields are declared in.
    """
    ready = []
    for name in FieldName:
        if state.fields[name].status != FieldStatus.MISSING:
            continue
        if all(state.fields[dep].status in RELIABLE for dep in DEPENDS_ON[name]):
            ready.append(name)
    return sorted(ready, key=lambda n: -BASE_WEIGHT[n])  # sorted() is stable


class CompletenessReport(BaseModel):
    complete: bool
    missing: list[FieldName]         # no value yet
    conflicting: list[FieldName]     # sources disagree
    not_confirmed: list[FieldName]   # has a value, but only known or uncertain
    low_confidence: list[FieldName]  # confirmed, but confidence is below the minimum


def check_completeness(state: Blueprint) -> CompletenessReport:
    """Is the plan defined well enough to start building?"""
    missing, conflicting, not_confirmed, low_confidence = [], [], [], []
    for name in REQUIRED_FIELDS:
        f = state.fields[name]
        if f.status == FieldStatus.MISSING:
            missing.append(name)
        elif f.status == FieldStatus.CONFLICTING:
            conflicting.append(name)
        elif f.status != FieldStatus.CONFIRMED:
            not_confirmed.append(name)
        elif f.confidence < MIN_CONFIDENCE:
            low_confidence.append(name)
    return CompletenessReport(
        complete=not (missing or conflicting or not_confirmed or low_confidence),
        missing=missing,
        conflicting=conflicting,
        not_confirmed=not_confirmed,
        low_confidence=low_confidence,
    )