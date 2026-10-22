"""The numbers that describe one field right now. Same code for training (simulated) and use (real).

    x = field_risk_features(state, events, "target_variable", days_left=20)   # None if no value yet

The first four are the state classifier's own numbers (state/features.py), reused as they are. The
rest come from the decision records (decisions/tracker.py) and the Domino graph (decisions/domino.py).
Time is measured in messages, not minutes: "staleness" is how many mentions of OTHER fields came
after this field was last mentioned, which is the same in a simulated and a real chat.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from ..common.schema import DEPENDS_ON, DecisionStatus, FieldName
from ..conflicts.same import same_value
from ..decisions.domino import descendants, direct_children
from ..decisions.tracker import active_for, current_decision, decisions_for
from ..state.features import EXTRACTOR_ACTOR, field_features

FEATURE_NAMES = [
    # the state classifier's numbers
    "log_source_count", "agreement", "latest_hedged", "distinct_speakers",
    # the decision record
    "is_settled", "is_finalized", "n_agreed", "n_objections", "has_reason", "times_changed",
    # time and the rest of the plan
    "log_staleness", "n_dependants", "related_disputed", "parent_changed_since",
    "deadline_known", "log_days_left", "plan_settled_share",
]
SETTLED = (DecisionStatus.CONFIRMED, DecisionStatus.FINALIZED)
NO_DEADLINE_DAYS = 60          # stand-in when no deadline is known (deadline_known says so)


def _kind(e) -> str:
    return getattr(e.event_type, "value", e.event_type)


def rivals(state, field: str) -> int:
    """Active decisions for the field whose value differs from the current one."""
    current = current_decision(state, field)
    if current is None:
        return 0
    return sum(1 for d in active_for(state, field)
               if d.decision_id != current.decision_id and not same_value(field, d.value, current.value))


def in_dispute(state, field: str) -> bool:
    return rivals(state, field) > 0


def staleness(events: list, field: str) -> int:
    """Mentions of other fields since this field was last mentioned."""
    count = 0
    for e in reversed(events):
        if _kind(e) != "field_updated" or e.actor != EXTRACTOR_ACTOR:
            continue
        if e.payload.get("field") == field:
            return count
        count += 1
    return count


def field_risk_features(state, events: Iterable, field: str, days_left: Optional[float] = None) -> Optional[list[float]]:
    """FEATURE_NAMES for one field, or None if nobody has given it a value yet."""
    events = sorted(events, key=lambda e: e.version)
    base = field_features(events, field)
    current = current_decision(state, field)
    if base is None or current is None:
        return None
    n, agreement, hedged, speakers = base
    history = decisions_for(state, field)
    changed = sum(1 for d in history if d.status in (DecisionStatus.SUPERSEDED, DecisionStatus.REJECTED))

    parents = [p.value for p in DEPENDS_ON[FieldName(field)]]
    related = parents + sorted(direct_children(field))
    since = current.created_at
    parent_changed = any(d.status == DecisionStatus.SUPERSEDED and d.updated_at > since
                         for p in parents for d in decisions_for(state, p))

    settled = sum(1 for name in FieldName
                  if (d := current_decision(state, name.value)) is not None and d.status in SETTLED)
    known = days_left is not None
    return [
        math.log1p(n),
        agreement,
        hedged,
        speakers,
        1.0 if current.status in SETTLED else 0.0,
        1.0 if current.status == DecisionStatus.FINALIZED else 0.0,
        float(len(current.agreed_by)),
        float(len(current.objected_by)),
        1.0 if current.rationale else 0.0,
        float(changed),
        math.log1p(staleness(events, field)),
        float(len(descendants(field))),
        1.0 if any(in_dispute(state, r) for r in related) else 0.0,
        1.0 if parent_changed else 0.0,
        1.0 if known else 0.0,
        math.log1p(max(0.0, days_left if known else NO_DEADLINE_DAYS)),
        settled / len(FieldName),
    ]
