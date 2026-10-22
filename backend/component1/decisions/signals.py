"""What Buddy is told about decision changes. Only the ones worth a word: a decision finalized or a
proposal rejected. Re-checks (pending_fix) reach Buddy through pipeline.state_signals."""
from __future__ import annotations

from typing import Iterable

from ..agent.signals import FINALIZED, SAVED, Signal
from .tracker import Change


def decision_signals(store, changes: Iterable[Change]) -> list[Signal]:
    out = []
    for c in changes:
        d = store.state.decisions.get(c.decision_id)
        if d is None:
            continue
        if c.kind == "finalized":
            who = d.proposed_by if not str(d.proposed_by).startswith("doc:") else "a document"
            detail = f"decision finalized, proposed by {who}" + (f", because {d.rationale}" if d.rationale else "")
            out.append(Signal(SAVED, c.field, (c.value,), detail=detail))
            out.append(Signal(FINALIZED, c.field))          # a calm moment: Buddy may ask its next question
        elif c.kind == "rejected":
            out.append(Signal(SAVED, c.field, (c.value,), detail=f"proposal rejected by {c.by}"))
    return out
