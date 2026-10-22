"""Pick the proposal a reply answers, among the recent ones.

    parent = link_reply(bundle, reply, recent_messages)          # a Message, or None

Candidates are the proposals among the last WINDOW messages (and within MAX_MINUTES). The model
scores each (reply, candidate) pair; the best one is chosen if its score is high enough.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from .features import Message, pair_features

log = logging.getLogger("component1")

MODEL_PATH = Path(__file__).parent / "models" / "reply_linker.joblib"
WINDOW = 15
MAX_MINUTES = 120
MIN_SCORE = 0.2


def candidates(reply: Message, history: list[Message]) -> list[Message]:
    recent = [m for m in history[-WINDOW:] if m.field and m.value]
    return [m for m in recent if (reply.seconds - m.seconds) / 60 <= MAX_MINUTES]


def pairs(reply: Message, history: list[Message], answered: Optional[dict[str, set]] = None):
    """[(candidate, features)] for every candidate proposal of this reply."""
    answered = answered or {}
    props = candidates(reply, history)
    out = []
    for cand in props:
        idx = next(i for i, m in enumerate(history) if m.message_id == cand.message_id)
        between = len(history) - 1 - idx
        out.append((cand, pair_features(reply, cand, props, between,
                                        reply.speaker in answered.get(cand.message_id, set()))))
    return out


def link_reply(bundle: Optional[dict], reply: Message, history: list[Message],
               answered: Optional[dict[str, set]] = None) -> Optional[Message]:
    found = pairs(reply, history, answered)
    if not found:
        return None
    if bundle is None:
        return found[-1][0]                       # the baseline: the newest proposal
    X = np.array([f for _, f in found])
    scores = bundle["model"].predict_proba(X)[:, list(bundle["model"].classes_).index(1)]
    best = int(np.argmax(scores))
    return found[best][0] if scores[best] >= MIN_SCORE else None


_default: Optional[dict] = None
_tried = False


def default_bundle() -> Optional[dict]:
    global _default, _tried
    if not _tried:
        _tried = True
        try:
            if MODEL_PATH.exists():
                _default = joblib.load(MODEL_PATH)
        except Exception:
            log.warning("Component 1 could not load the reply linker", exc_info=True)
    return _default
