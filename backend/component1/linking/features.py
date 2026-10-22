"""Describe one (reply, candidate proposal) pair with a few numbers. Same code for training and use.

A Message is what the linker knows about a chat message: who, when, its text, and, for proposals,
the field and value it proposed.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from ..state.normalize import normalize_value

FEATURE_NAMES = [
    "log_messages_between", "log_minutes_between", "is_latest_proposal", "proposal_rank",
    "same_speaker", "names_the_value", "names_the_field", "names_another_value", "names_another_field",
    "word_overlap", "already_answered_by_replier",
]

FIELD_WORDS = {
    "algorithm": ("algorithm", "model", "method", "approach"),
    "evaluation_metric": ("metric", "measure", "score", "evaluate", "evaluation"),
    "dataset": ("dataset", "data", "data set"),
    "target_variable": ("target", "predict", "label", "output"),
    "task_type": ("task", "classification", "regression", "forecasting"),
    "problem": ("problem",),
    "domain": ("domain", "area", "sector"),
    "objective": ("goal", "objective", "aim"),
    "resources": ("gpu", "compute", "colab", "cloud", "resources"),
    "timeline": ("deadline", "timeline", "weeks", "due"),
    "dataset_size": ("rows", "records", "size", "samples"),
}
_STOP = set("the a an and or to of for is it its we our us i you this that be on in with use let's lets go "
            "yes no ok okay sure agreed agree sounds good fine think".split())
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Message:
    message_id: str
    speaker: str
    seconds: float
    text: str
    field: Optional[str] = None      # set for proposals
    value: Optional[str] = None


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 1}


def _plain(text: str) -> str:
    """Normalised with punctuation as spaces, so "arrivals," loses its plural "s" like "arrivals" does."""
    return normalize_value(re.sub(r"[^\w\s]", " ", text or ""))


def _mentions(text: str, value: Optional[str]) -> bool:
    if not value:
        return False
    v = _plain(value)
    return bool(v) and re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", _plain(text)) is not None


def pair_features(reply: Message, candidate: Message, proposals_before: list[Message],
                  messages_between: int, answered_by_replier: bool) -> list[float]:
    """proposals_before: the candidate proposals in time order (newest last), including this one."""
    rank = len(proposals_before) - 1 - proposals_before.index(candidate)
    minutes = max(0.0, reply.seconds - candidate.seconds) / 60
    others = [p for p in proposals_before if p is not candidate]
    text = (reply.text or "").lower()

    def names_field(p: Message) -> bool:
        return any(w in text for w in FIELD_WORDS.get(p.field or "", ()))

    return [
        math.log1p(messages_between),
        math.log1p(minutes),
        1.0 if rank == 0 else 0.0,
        float(rank),
        1.0 if reply.speaker == candidate.speaker else 0.0,
        1.0 if _mentions(reply.text, candidate.value) else 0.0,
        1.0 if names_field(candidate) else 0.0,
        1.0 if any(_mentions(reply.text, p.value) for p in others) else 0.0,
        # The reply names a DIFFERENT proposal's field ("for the metric" while this one is the model).
        1.0 if any(names_field(p) and p.field != candidate.field for p in others) and not names_field(candidate)
        else 0.0,
        len(_words(reply.text) & _words(candidate.text)) / max(1, len(_words(reply.text))),
        1.0 if answered_by_replier else 0.0,
    ]
