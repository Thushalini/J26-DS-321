"""Conflict risk: how likely each field is to be disputed or changed soon, and why.

    risks = field_risks(bundle, state, events, days_left=20)
    risks["target_variable"] == {"risk": 0.31, "reasons": ["only one person has mentioned it", ...]}

Fields with no value, and fields already in a disagreement (Buddy already asks about those), get no
score. The reasons are the facts that raise this field's risk the most: each fact is set to what is
typical for a field that did NOT change (the bundle's "safe_reference"), and the bigger the drop in
risk, the stronger the reason. Trained on SIMULATED histories (train_risk.py); not calibrated on real teams.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np

from ..common.schema import FieldName
from .features import FEATURE_NAMES, field_risk_features, in_dispute

log = logging.getLogger("component1")

MODEL_PATH = Path(__file__).parent / "models" / "conflict_risk.joblib"
MIN_REASON_DROP = 0.02        # a fact must move the risk by at least this much to be named as a reason
MAX_REASONS = 2

# What each fact means in words, when it points towards risk. None: never named as a reason.
REASON_TEXT = {
    "log_source_count": lambda x: "it has only been mentioned once" if x <= np.log1p(1) else None,
    "agreement": lambda x: "people have given different values for it before" if x < 1 else None,
    "latest_hedged": lambda x: "the last mention sounded unsure" if x >= 1 else None,
    "distinct_speakers": lambda x: "only one person has mentioned it" if x <= 1 else None,
    "is_settled": lambda x: "nobody else has agreed with it yet" if x < 1 else None,
    "n_agreed": lambda x: "nobody else has agreed with it yet" if x < 1 else None,
    "n_objections": lambda x: "someone has objected to it" if x >= 1 else None,
    "has_reason": lambda x: "no reason was given for it" if x < 1 else None,
    "times_changed": lambda x: "it has changed before" if x >= 1 else None,
    "log_staleness": lambda x: "it has not been discussed for a while" if x >= np.log1p(10) else None,
    "related_disputed": lambda x: "a related field is disputed" if x >= 1 else None,
    "parent_changed_since": lambda x: "a field it depends on has changed since" if x >= 1 else None,
    "plan_settled_share": lambda x: "most of the plan is still open" if x < 0.5 else None,
}


def risk_of(bundle: dict, x: list[float]) -> float:
    return float(bundle["model"].predict_proba(np.array([x]))[0, 1])


def reasons_for(bundle: dict, x: list[float]) -> list[str]:
    base = risk_of(bundle, x)
    safe = bundle["safe_reference"]
    found: dict[str, float] = {}
    for i, name in enumerate(FEATURE_NAMES):
        text_of = REASON_TEXT.get(name)
        text = text_of(x[i]) if text_of else None
        if text is None or x[i] == safe[i]:
            continue
        changed = list(x)
        changed[i] = safe[i]
        drop = base - risk_of(bundle, changed)
        if drop >= MIN_REASON_DROP:
            found[text] = max(drop, found.get(text, 0.0))       # two facts can share one sentence
    return [t for t, _ in sorted(found.items(), key=lambda kv: -kv[1])][:MAX_REASONS]


def field_risks(bundle: Optional[dict], state, events: Iterable, days_left: Optional[float] = None,
                with_reasons: bool = True) -> dict:
    """field -> {"risk", "reasons"} for every field that has a value and is not already disputed.
    with_reasons=False skips the explanations (they cost one extra prediction per fact)."""
    if bundle is None:
        return {}
    events = list(events)
    out = {}
    for name in FieldName:
        field = name.value
        if in_dispute(state, field):
            continue
        x = field_risk_features(state, events, field, days_left)
        if x is None:
            continue
        out[field] = {"risk": round(risk_of(bundle, x), 3), "reasons": reasons_for(bundle, x) if with_reasons else []}
    return out


_default: Optional[dict] = None
_tried = False


def default_bundle() -> Optional[dict]:
    """The saved model, loaded once; None if it has not been trained (then Buddy behaves as before)."""
    global _default, _tried
    if not _tried:
        _tried = True
        try:
            if MODEL_PATH.exists():
                _default = joblib.load(MODEL_PATH)
        except Exception:
            log.warning("Component 1 could not load the conflict-risk model", exc_info=True)
    return _default
