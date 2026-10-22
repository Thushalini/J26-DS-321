"""Simulated planning histories where it is known which fields are disputed or changed later.

Each team discusses the plan one message at a time. Every field has a hidden "real disagreement"
level: the higher it is, the more people hedge, the fewer agree, and the more likely someone later
proposes a different value. The model never sees that hidden level, only what a real chat would
show (state/features.py and the decision records), so it cannot be perfect.

Messages go through Buddy's real decision functions (decisions/tracker.py), so the histories have
exactly the shape of real ones. The probabilities below are assumptions, listed for the report.

    X, y, groups = generate_dataset(n_teams=600, seed=1)    # one row per (field, moment); groups = team
"""
from __future__ import annotations

import os

import numpy as np

from ..common.events import EventStore
from ..common.schema import DEPENDS_ON, DecisionStatus, FieldName, FieldStatus, Provenance, SourceType
from ..decisions.tracker import current_decision, decisions_for, find_active, on_reply, on_statement
from ..state.features import EXTRACTOR_ACTOR
from .features import field_risk_features, in_dispute

VALUES = {
    "problem": ["predicting tourist arrivals", "detecting fraud in payments", "forecasting hotel demand"],
    "domain": ["tourism", "finance", "healthcare"],
    "objective": ["help hotels plan staff", "reduce fraud losses", "improve demand planning"],
    "dataset": ["the SLTDA arrivals data", "the Kaggle hotel bookings data", "the UCI credit data"],
    "target_variable": ["tourist arrivals", "hotel occupancy", "fraud label"],
    "task_type": ["regression", "classification", "forecasting"],
    "algorithm": ["XGBoost", "Random Forest", "LSTM", "Logistic Regression", "ARIMA"],
    "evaluation_metric": ["RMSE", "MAE", "F1-score", "accuracy"],
    "resources": ["Google Colab", "a laptop", "AWS"],
    "dataset_size": ["500 rows", "5000 rows", "50000 rows"],
}
FIELDS = list(VALUES)                 # the timeline is not simulated: it is used as days_left instead
REASONS = ["it fits the data", "the classes are imbalanced", "it is easy to explain", "we have little time"]

ASSUMPTIONS = {
    "disagreement": (1.2, 2.5),             # Beta(a, b): hidden disagreement per field
    "steps": (50, 110),                     # messages per team
    "days_per_step": 0.4,
    "deadline_known": 0.6,
    "hedge": (0.15, 0.45),                  # p(hedged) = a + b * disagreement
    "reason": (0.2, 0.5),                   # p(reason) = a + b * team carefulness
    "agree": (0.10, 0.70),                  # p(agree | no change) = a + b * (1 - disagreement)
    "change": (0.03, 0.30),                 # p(someone proposes another value) = a + b * disagreement
    "change_if_settled": 0.4,               # ... times this once the value is confirmed or finalized
    "change_if_parent_moved": 2.0,          # ... times this if a field it depends on changed or is disputed
    "horizon": 15,                          # label: a different value is proposed within this many messages
    "snapshot_every": 4,
}


def _mention(store, field, value, speaker, step, hedged):
    prov = Provenance(source_type=SourceType.CHAT, source_id=f"s{step}", speaker=speaker, text_span=value)
    store.update_field(FieldName(field), value, FieldStatus.UNCERTAIN if hedged else FieldStatus.KNOWN,
                       0.6, prov, EXTRACTOR_ACTOR)


# store.state builds a fresh copy of the blueprint on every read, so each step reads it once ("st").
def _ready(st, field) -> bool:
    return all(current_decision(st, p.value) is not None for p in DEPENDS_ON[FieldName(field)]
               if p.value in VALUES)


def _settled(st, field) -> bool:
    cur = current_decision(st, field)
    return cur is not None and cur.status in (DecisionStatus.CONFIRMED, DecisionStatus.FINALIZED)


def _parent_moved(st, field) -> bool:
    cur = current_decision(st, field)
    for p in DEPENDS_ON[FieldName(field)]:
        if p.value not in VALUES:
            continue
        if in_dispute(st, p.value):
            return True
        if cur and any(d.status == DecisionStatus.SUPERSEDED and d.updated_at > cur.created_at
                       for d in decisions_for(st, p.value)):
            return True
    return False


def simulate_team(rng, a: dict = ASSUMPTIONS):
    """One team's planning history. Returns (rows, labels): rows are (features, step, field)."""
    os.environ["C1_NLI"] = "off"                 # values here differ by name; no meaning model needed
    store = EventStore("sim")
    speakers = [f"M-{i}" for i in range(1, int(rng.integers(2, 6)) + 1)]
    hidden = {f: float(rng.beta(*a["disagreement"])) for f in FIELDS}
    careful = float(rng.random())
    steps = int(rng.integers(*a["steps"]))
    days0 = float(rng.uniform(10, 90))
    deadline_known = rng.random() < a["deadline_known"]
    new_values: list[tuple[int, str]] = []       # (step, field) whenever a NEW value is proposed
    rows = []

    def say(field, value, speaker, step, hedged, intent=None):
        why = f" because {rng.choice(REASONS)}" if rng.random() < a["reason"][0] + a["reason"][1] * careful else ""
        _mention(store, field, value, speaker, step, hedged)
        if find_active(st, field, value) is None:
            new_values.append((step, field))
        on_statement(store, field, value, speaker, f"s{step}", f"Let's use {value}{why}.", intent)

    for step in range(1, steps + 1):
        days_left = max(0.0, days0 - step * a["days_per_step"])
        st = store.state
        open_fields = [f for f in FIELDS if current_decision(st, f) is None and _ready(st, f)]
        started = [f for f in FIELDS if current_decision(st, f) is not None]
        weights = [3.0] * len(open_fields) + [2.0 if not _settled(st, f) else 0.5 for f in started]
        choices = open_fields + started
        if not choices:
            continue
        field = choices[int(rng.choice(len(choices), p=np.array(weights) / sum(weights)))]
        d = hidden[field]
        speaker = str(rng.choice(speakers))
        others = [s for s in speakers if s != speaker] or speakers
        hedged = rng.random() < a["hedge"][0] + a["hedge"][1] * d
        cur = current_decision(st, field)

        if cur is None:                                           # first value for the field
            say(field, str(rng.choice(VALUES[field])), speaker, step, hedged)
        elif in_dispute(st, field):                              # settle the disagreement
            options = [x for x in st.decisions.values()
                       if x.field.value == field and x.status.value in ("proposed", "confirmed", "finalized")]
            pick = options[-1] if rng.random() < 0.5 else options[int(rng.integers(len(options)))]
            backer = str(rng.choice([s for s in speakers if s != pick.proposed_by] or speakers))
            say(field, pick.value, backer, step, False, intent="agreement")
        else:
            p_change = a["change"][0] + a["change"][1] * d
            p_change *= a["change_if_settled"] if _settled(st, field) else 1.0
            p_change *= a["change_if_parent_moved"] if _parent_moved(st, field) else 1.0
            p_change *= 0.5 + 0.5 * min(1.0, days_left / 30)
            if rng.random() < p_change:                           # someone wants something else
                alternatives = [v for v in VALUES[field] if v != cur.value]
                say(field, str(rng.choice(alternatives)), str(rng.choice(others)), step, hedged)
            elif rng.random() < a["agree"][0] + a["agree"][1] * (1 - d):
                backer = str(rng.choice([s for s in speakers if s != cur.proposed_by] or speakers))
                finalize = cur.status.value == "confirmed" and rng.random() < 0.3 + 0.4 * careful
                intent = "finalization" if finalize else "agreement"
                if rng.random() < 0.5:
                    say(field, cur.value, backer, step, hedged, intent=intent)
                else:
                    on_reply(store, cur, backer, intent)
            elif rng.random() < d:                                # doubt without an alternative
                on_reply(store, cur, str(rng.choice(others)), "rejection")
            else:                                                 # a repeat, unsure more often if disputed
                _mention(store, field, cur.value, str(rng.choice(others)), step, hedged)

        if step % a["snapshot_every"] == 0 and step >= 8:
            st = store.state
            for f in FIELDS:
                if in_dispute(st, f):
                    continue                                      # already a conflict: Buddy handles it
                x = field_risk_features(st, store.events, f, days_left if deadline_known else None)
                if x is not None:
                    rows.append((x, step, f))

    labels = [int(any(f == g and step < s <= step + a["horizon"] for s, g in new_values)) for _, step, f in rows]
    keep = [i for i, (_, step, _) in enumerate(rows) if step + a["horizon"] <= steps]   # full horizon only
    return [rows[i] for i in keep], [labels[i] for i in keep]


def generate_dataset(n_teams: int = 600, seed: int = 1):
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for team in range(n_teams):
        rows, labels = simulate_team(rng)
        X += [r[0] for r in rows]
        y += labels
        groups += [team] * len(rows)
    return np.array(X, dtype=float), np.array(y), np.array(groups)
