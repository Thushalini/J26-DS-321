"""Build the four state-classifier numbers for a field from the event log.

Only the extractor's own FIELD_UPDATED events count as "mentions". Events written by
other parts (for example the classifier recording a status) are ignored, so they never
count as extra evidence.

The numbers match the ones the classifier was trained on:
    source_count       how many times the field was mentioned
    agreement          share of mentions that carry the most common (normalised) value
    latest_hedged      1 if the most recent mention sounded unsure, else 0
    distinct_speakers  how many different people mentioned it

Known limit: a bare "I agree" is not an extraction, so it is not a mention yet.
That needs the decision-intent classifier (features 8 and 9).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

from .normalize import normalize_value

FEATURE_ORDER = ["source_count", "agreement", "latest_hedged", "distinct_speakers"]
EXTRACTOR_ACTOR = "extractor"


def _plain(x):
    """Accept an Enum or a plain string."""
    return getattr(x, "value", x)


def mentions_for_field(events: Iterable, field_name) -> list[dict]:
    field_name = _plain(field_name)
    mentions = []
    for e in sorted(events, key=lambda e: getattr(e, "version", 0)):
        if _plain(e.event_type) != "field_updated" or e.actor != EXTRACTOR_ACTOR:
            continue
        p = e.payload
        if p.get("field") != field_name:
            continue
        provenance = p.get("provenance") or {}
        mentions.append({
            "value": normalize_value(p["new_value"]),
            "speaker": provenance.get("speaker"),
            "hedged": p.get("new_status") == "uncertain",
        })
    return mentions


def field_features(events: Iterable, field_name) -> Optional[list[float]]:
    """The four numbers in FEATURE_ORDER, or None if the field was never mentioned."""
    mentions = mentions_for_field(events, field_name)
    if not mentions:
        return None
    n = len(mentions)
    counts = Counter(m["value"] for m in mentions)
    return [
        float(n),
        max(counts.values()) / n,
        1.0 if mentions[-1]["hedged"] else 0.0,
        float(len({m["speaker"] for m in mentions})),
    ]