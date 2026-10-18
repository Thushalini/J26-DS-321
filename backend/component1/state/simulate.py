"""Synthetic training data for the context state classifier (feature 4).

Why simulate conversations instead of writing "if agreement is high, label it
confirmed"? That would teach the model your own rule back to you. Here the label
comes from the codebook (see labeling/labeling_guide.md, Task 2), applied to a
simulated conversation about ONE field. The model only sees a few summary numbers,
not the hidden details (who agreed, who sounded unsure), so it cannot be perfect.

"missing" is not predicted. It is a plain rule: no value has been stated yet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LABELS = ["known", "uncertain", "conflicting", "confirmed"]
# The first four are the features named in the proposal. The last two are extra signals the
# pipeline really has: the extractor's "hedged" flag, and who said what (provenance).
FEATURE_NAMES = [
    "source_count", "agreement", "mean_confidence", "recency_minutes",
    "latest_hedged", "distinct_speakers",
]
PROPOSAL_FEATURES = FEATURE_NAMES[:4]


@dataclass
class Mention:
    kind: str        # "state" (says a value), "endorse" (agrees with it), "repeat" (same person says it again)
    speaker: int     # 0, 1 or 2
    value: str       # which value this mention supports
    hedged: bool     # sounded unsure ("maybe", "I think")
    confidence: float


def _confidence(rng, hedged: bool, base: float) -> float:
    c = rng.normal(base - (0.15 if hedged else 0.0), 0.10)
    return round(float(np.clip(c, 0.05, 0.99)), 3)


def _other_speaker(rng, speaker: int) -> int:
    return int(rng.choice([s for s in (0, 1, 2) if s != speaker]))


def simulate_mentions(rng, hedge_rate: float = 0.30, conf_base: float = 0.85) -> list[Mention]:
    """One simulated conversation about one field, in order."""
    first_hedged = bool(rng.random() < hedge_rate)
    mentions = [Mention("state", 0, "A", first_hedged, _confidence(rng, first_hedged, conf_base))]
    extra = int(rng.choice([0, 1, 2, 3], p=[0.35, 0.30, 0.20, 0.15]))
    for _ in range(extra):
        latest = next(m for m in reversed(mentions) if m.kind == "state")
        kind = str(rng.choice(["endorse", "contradict", "repeat"], p=[0.40, 0.30, 0.30]))
        if kind == "endorse":
            mentions.append(Mention("endorse", _other_speaker(rng, latest.speaker), latest.value,
                                    False, _confidence(rng, False, conf_base)))
        elif kind == "contradict":
            hedged = bool(rng.random() < hedge_rate)
            other_value = "B" if latest.value == "A" else "A"
            mentions.append(Mention("state", _other_speaker(rng, latest.speaker), other_value,
                                    hedged, _confidence(rng, hedged, conf_base)))
        else:  # the same person says it again. This is NOT confirmation.
            mentions.append(Mention("repeat", latest.speaker, latest.value,
                                    False, _confidence(rng, False, conf_base)))
    return mentions


def label_from_mentions(mentions: list[Mention]) -> str:
    """The codebook rules (Task 2), applied to the end of the conversation."""
    last_idx = max(i for i, m in enumerate(mentions) if m.kind == "state")
    last_state = mentions[last_idx]
    values = {m.value for m in mentions if m.kind == "state"}
    endorsed = any(
        m.kind == "endorse" and i > last_idx and m.speaker != last_state.speaker
        for i, m in enumerate(mentions)
    )
    if len(values) > 1 and not endorsed:
        return "conflicting"
    if endorsed:
        return "confirmed"
    if last_state.hedged:
        return "uncertain"
    return "known"


def features_from_mentions(mentions: list[Mention], recency_minutes: float) -> list[float]:
    counts: dict[str, int] = {}
    for m in mentions:
        counts[m.value] = counts.get(m.value, 0) + 1
    n = len(mentions)
    latest_state = next(m for m in reversed(mentions) if m.kind == "state")
    return [
        float(n),                                               # source_count
        max(counts.values()) / n,                               # agreement: share backing the leading value
        float(np.mean([m.confidence for m in mentions])),       # mean_confidence
        float(recency_minutes),                                 # recency_minutes
        1.0 if latest_state.hedged else 0.0,                    # latest_hedged
        float(len({m.speaker for m in mentions})),              # distinct_speakers
    ]


def generate_dataset(
    n: int,
    seed: int = 42,
    label_noise: float = 0.05,
    hedge_rate: float = 0.30,
    conf_base: float = 0.85,
):
    """Return (X, y). X has the FEATURE_NAMES columns, y holds the labels.

    label_noise: share of labels flipped to a different class, a stand-in for
    human labelers disagreeing (Kappa is never 1.0).
    recency_minutes is random and has NO effect on the label by construction,
    so on this data the model should learn to ignore it. Real data may differ.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        mentions = simulate_mentions(rng, hedge_rate, conf_base)
        label = label_from_mentions(mentions)
        if rng.random() < label_noise:
            label = str(rng.choice([l for l in LABELS if l != label]))
        X.append(features_from_mentions(mentions, rng.integers(0, 600)))
        y.append(label)
    return np.array(X), np.array(y)