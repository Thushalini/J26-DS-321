"""Buddy's speaking rules: when the assistant stays silent and when it speaks.

Pure Python. No database, no model, no web code, so it is easy to test.
Times are plain seconds, passed in by the caller.

How it is used:
    buddy = Buddy()
    buddy.heard_message(now)               # call whenever a person sends a message
    replies = buddy.step(signals, now)     # call after the pipeline runs, and also on a timer

step() returns an empty list, or exactly ONE message. A message can hold a short
"Noted: ..." line followed by one question or warning, so Buddy never sends two
messages in a row for the same event.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .signals import CHECK, CLARIFY, CONFIRM, CONFLICT, FINALIZED, FOUND, GAP, SAVED, Signal

FEASIBILITY_PREFIX = "whether the plan is feasible depends on"   # active/sensitivity.py starts its sentences so


@dataclass(frozen=True)
class BuddyConfig:
    # Starting values. Tune them by testing with real people.
    min_seconds_between_questions: float = 120.0
    quiet_seconds_before_gap_question: float = 90.0
    repeat_after_seconds: float = 1800.0


def field_label(field: str) -> str:
    return field.replace("_", " ")


def saved_line(signals: list[Signal]) -> str:
    items = []
    for s in signals:
        value = s.values[0] if s.values else ""
        notes = []
        if s.source:
            notes.append(f"from {s.source}")
        if s.detail:
            notes.append(s.detail)
        suffix = f" ({', '.join(notes)})" if notes else ""
        items.append(f"{field_label(s.field)} = {value}{suffix}")
    return "Noted: " + "; ".join(items) + "."


def conflict_text(signal: Signal) -> str:
    label = field_label(signal.field)
    values = list(signal.values)
    if len(values) < 2:
        return f"I saw conflicting information about the {label}. Which value should we go with?"
    if len(values) == 2:
        joined = f"{values[0]} and {values[1]}"
    else:
        joined = ", ".join(values[:-1]) + f" and {values[-1]}"
    return f"I saw different answers for the {label}: {joined}. Which one should we go with?"


def check_text(signal: Signal) -> str:
    source = f" (source: {signal.source})" if signal.source else ""
    return f"Possible problem with the {field_label(signal.field)}: {signal.detail}{source} Is that intended?"


def found_text(signal: Signal) -> str:
    value = signal.values[0] if signal.values else ""
    where = signal.source or "the uploaded documents"
    return (f"I found a possible {field_label(signal.field)} in {where}: {value}. Should we use that? "
            f"(You can also accept or dismiss it on the Blueprint tab.)")


def gap_text(signal: Signal) -> str:
    text = _gap_question(signal)
    if signal.detail.startswith(FEASIBILITY_PREFIX):          # the answer decides whether the plan can work
        text += f" It matters because {signal.detail}."
    return text


def _gap_question(signal: Signal) -> str:
    if signal.field == "dataset_size":
        return "I noticed nobody has said how big the dataset is. Roughly how many rows does it have?"
    if signal.field == "timeline":
        return ("I noticed there is no decision yet on the timeline. When is the deadline, "
                "or how many weeks does the team have?")
    return f"I noticed there is no decision yet on the {field_label(signal.field)}. Could someone confirm what it will be?"


def confirm_text(signal: Signal) -> str:
    value = signal.values[0] if signal.values else ""
    reasons = [r for r in signal.detail.split("; ") if r]
    why = " and ".join(reasons) if reasons else "it may still change"
    return f"Can someone confirm the {field_label(signal.field)}: {value}? I ask because {why}."


def clarify_text(signal: Signal) -> str:
    return f"A detail is missing: {signal.detail}"


ASK_TEXT = {GAP: gap_text, CONFIRM: confirm_text, CLARIFY: clarify_text}


class Buddy:
    def __init__(self, config: BuddyConfig = BuddyConfig()):
        self.config = config
        self._said: dict[tuple, float] = {}
        self._last_question_at: Optional[float] = None
        self._last_human_message_at: Optional[float] = None
        self.last_step_keys: list[tuple] = []     # what the last step() said, so it can be saved
        self.last_step_asked: bool = False        # True if the last step() asked a question or gave a warning

    def restore(self, keys, at: float, asked: bool) -> None:
        """Remember something said before a restart, so Buddy does not repeat it."""
        for key in keys:
            self._said[tuple(key)] = at
        if asked:
            self._last_question_at = at

    def heard_message(self, now: float) -> None:
        self._last_human_message_at = now

    def step(self, signals: list[Signal], now: float) -> list[str]:
        """Return what Buddy says now: an empty list, or exactly one message."""
        saved = self._unsaid([s for s in signals if s.kind == SAVED], now)
        question = self._pick_question(signals, now)

        parts = []
        keys: list[tuple] = []
        if saved:
            for s in saved:
                self._said[s.key] = now
                keys.append(s.key)
            parts.append(saved_line(saved))
        if question is not None:
            signal, text = question
            self._said[signal.key] = now
            self._last_question_at = now
            keys.append(signal.key)
            parts.append(text)
        self.last_step_keys = keys
        self.last_step_asked = question is not None
        return ["\n".join(parts)] if parts else []

    # ------------------------------------------------------------ helpers
    def _pick_question(self, signals: list[Signal], now: float):
        """At most one question or warning, most urgent first."""
        for kind, make_text in ((CONFLICT, conflict_text), (CHECK, check_text)):
            s = self._first_unsaid([x for x in signals if x.kind == kind], now)
            if s is not None:
                return s, make_text(s)

        if not (self._calm_enough(signals, now) and self._rate_ok(now)):
            return None

        found = self._first_unsaid([x for x in signals if x.kind == FOUND], now)
        if found is not None:
            return found, found_text(found)

        # Gap, confirm and clarify questions share the last slot. Scored ones (active/priority.py) go by
        # priority; unscored ones keep the order they came in (the old fixed order, most important gap first).
        found_fields = {x.field for x in signals if x.kind == FOUND}
        asks = [x for x in signals if x.kind in ASK_TEXT and x.field not in found_fields]
        if any(x.priority for x in asks):
            asks = sorted(asks, key=lambda x: -x.priority)
        ask = self._first_unsaid(asks, now)
        if ask is not None:
            return ask, ASK_TEXT[ask.kind](ask)
        return None

    def _is_new(self, signal: Signal, now: float) -> bool:
        last = self._said.get(signal.key)
        return last is None or now - last >= self.config.repeat_after_seconds

    def _first_unsaid(self, signals: list[Signal], now: float) -> Optional[Signal]:
        return next((s for s in signals if self._is_new(s, now)), None)

    def _unsaid(self, signals: list[Signal], now: float) -> list[Signal]:
        return [s for s in signals if self._is_new(s, now)]

    def _rate_ok(self, now: float) -> bool:
        return (
            self._last_question_at is None
            or now - self._last_question_at >= self.config.min_seconds_between_questions
        )

    def _calm_enough(self, signals: list[Signal], now: float) -> bool:
        if any(s.kind == FINALIZED for s in signals):
            return True
        last = self._last_human_message_at
        return last is None or now - last >= self.config.quiet_seconds_before_gap_question