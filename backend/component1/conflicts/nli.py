"""The NLI model: does statement A contradict, imply, or say nothing about statement B?

    p = relation("The target is tourist arrivals.", "The target is hotel occupancy.")
    p == {"contradiction": 1.0, "entailment": 0.0, "neutral": 0.0}

Model: cross-encoder/nli-deberta-v3-small (about 180 MB, downloaded once), used as it is (no
training). Results are cached, so a pair is only checked once per process.
Switch it off with C1_NLI=off in backend/.env; then only known names and spellings are compared.
"""
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Callable, Optional

log = logging.getLogger("component1")

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
Relation = Callable[[str, str], Optional[dict]]

_model = None
_tried = False
_lock = threading.Lock()
_override: Optional[Relation] = None


def enabled() -> bool:
    return (os.environ.get("C1_NLI") or "on").strip().lower() not in ("off", "0", "false", "no")


def set_relation(fn: Optional[Relation]) -> None:
    """Replace the model (tests use a pretend one). None goes back to the real model."""
    global _override
    _override = fn
    _cached.cache_clear()


def _load():
    global _model, _tried
    with _lock:
        if not _tried:
            _tried = True
            try:
                from sentence_transformers import CrossEncoder
                _model = CrossEncoder(MODEL_NAME)
                log.info("Component 1 loaded the NLI model %s", MODEL_NAME)
            except Exception:
                log.warning("Component 1 could not load the NLI model; comparing names only", exc_info=True)
        return _model


@lru_cache(maxsize=4096)
def _cached(premise: str, hypothesis: str) -> Optional[tuple]:
    if _override is not None:
        out = _override(premise, hypothesis)
        return None if out is None else tuple(sorted(out.items()))
    model = _load()
    if model is None:
        return None
    scores = model.predict([(premise, hypothesis)], apply_softmax=True)[0]
    labels = model.model.config.id2label
    return tuple(sorted((labels[i].lower(), float(s)) for i, s in enumerate(scores)))


def relation(premise: str, hypothesis: str) -> Optional[dict]:
    """{"contradiction", "entailment", "neutral"} probabilities, or None if NLI is off or unavailable."""
    if _override is None and not enabled():
        return None
    try:
        found = _cached(premise, hypothesis)
    except Exception:
        log.warning("Component 1 NLI check failed", exc_info=True)
        return None
    return dict(found) if found is not None else None
