"""The connected flow for ONE chat message.

    message -> extract facts (feature 3) -> judge how settled each touched field is
    (feature 4) -> record that judgement -> work out what Buddy should be told.

Nothing here talks to a database or the internet by itself. The language model is a
plain function you pass in, so tests can use a pretend one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .agent.signals import CONFLICT, GAP, SAVED, Signal
from .common.events import EventStore
from .common.schema import FieldName, FieldStatus
from .extraction.extractor import LLMFn, apply_message
from .gaps.gaps import ready_gaps
from .state.features import field_features, mentions_for_field
from .state.predict import predict_status

STATE_ACTOR = "state_classifier"

# Short notes Buddy adds to its "Noted: ..." line.
_SAVED_DETAIL = {
    "uncertain": "marked uncertain",
    "conflicting": "conflicts with an earlier value",
    "confirmed": "now confirmed",
    "known": "",
}


@dataclass
class MessageResult:
    saved: list[tuple[str, str]]                 # (field, value) the extractor saved from this message
    statuses: dict[str, tuple[str, float]]       # field -> (status, model confidence) for touched fields
    signals: list[Signal]                        # what Buddy should be told


def _conflict_values(store: EventStore, field: str) -> tuple[str, ...]:
    """The different values people gave, in the order they first appeared (original spelling)."""
    first_spelling: dict[str, str] = {}
    for mention in mentions_for_field(store.events, field):
        first_spelling.setdefault(mention["value"], mention["original"])
    return tuple(first_spelling.values())


def state_signals(store: EventStore) -> list[Signal]:
    """Signals that come from the blueprint as it stands now: conflicts first, then gaps.

    Used after every message, and also on a timer, so Buddy can speak up when the chat
    has gone quiet even though no new message arrived.
    """
    signals: list[Signal] = []
    state = store.state
    for name in FieldName:
        if state.fields[name].status == FieldStatus.CONFLICTING:
            signals.append(Signal(CONFLICT, name.value, _conflict_values(store, name.value)))
    for name in ready_gaps(state):
        signals.append(Signal(GAP, name.value))
    return signals


def process_message(
    store: EventStore,
    bundle: dict,
    llm: LLMFn,
    message_id: str,
    speaker: str,
    text: str,
    timestamp: Optional[datetime] = None,
) -> MessageResult:
    extraction_events = apply_message(store, message_id, speaker, text, llm, timestamp)

    # 1. Judge every field this message touched, and record the judgement.
    touched = list(dict.fromkeys(e.payload["field"] for e in extraction_events))
    statuses: dict[str, tuple[str, float]] = {}
    for field in touched:
        label, probability = predict_status(bundle, field_features(store.events, field))
        name = FieldName(field)
        current_value = store.state.fields[name].value
        # No new evidence is attached: this event only records the judgement.
        store.update_field(name, current_value, FieldStatus(label), probability, None, STATE_ACTOR)
        statuses[field] = (label, probability)

    # 2. Work out what Buddy should be told.
    signals: list[Signal] = []
    for event in extraction_events:
        field, value = event.payload["field"], event.payload["new_value"]
        signals.append(Signal(SAVED, field, (value,), detail=_SAVED_DETAIL.get(statuses[field][0], ""), source=speaker))

    signals.extend(state_signals(store))

    return MessageResult(
        saved=[(e.payload["field"], e.payload["new_value"]) for e in extraction_events],
        statuses=statuses,
        signals=signals,
    )