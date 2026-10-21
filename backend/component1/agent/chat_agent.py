"""One team's assistant: hears each chat message and decides what, if anything, to say.

It holds the blueprint (an EventStore) and Buddy's speaking rules for ONE team. It
knows nothing about databases, web servers or the chat screen, so it can be tested with
plain Python. Later a thin layer rebuilds it from saved events and posts its replies.

    agent = ChatAgent("team_1", bundle, llm)
    replies = agent.on_message("M1", "Member_A", "I think we should predict arrivals", now=0)
    replies = agent.tick(now=300)    # call on a timer, so Buddy can speak after a quiet spell

If an intent model is given, a message that backs up a value someone ELSE just proposed
("ok, lock that in") counts as one more person agreeing with that value.

If the agent is rebuilt from saved events (after a restart), the last saved mention is used as
the topic again, but only for the first message that follows.

Document suggestions (documents/candidates.py) are kept in `candidates`, field -> best one.
Buddy asks about a suggestion for a field that is still missing. If a team member then backs
it up ("yes, use that"), the suggestion is accepted, exactly as with the Blueprint tab's
Accept button. Accepted suggestions collect in `accepted` until the caller has saved them.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..common.events import EventStore
from ..documents.candidates import Candidate
from ..documents.integrate import accept_candidate, document_speaker
from ..extraction.extractor import LLMFn
from ..pipeline import _SAVED_DETAIL, MessageResult, process_message, state_signals
from ..state.features import EXTRACTOR_ACTOR
from .policy import Buddy
from .signals import FOUND, SAVED, Signal

log = logging.getLogger("component1")

ENDORSING_INTENTS = ("agreement", "finalization")
MIN_INTENT_CONFIDENCE = 0.5      # below this the intent label is ignored
TOPIC_LIFETIME = 6               # messages after which a proposed value is no longer "the topic"


class ChatAgent:
    def __init__(
        self,
        project_id: str,
        bundle: dict,
        llm: LLMFn,
        store: Optional[EventStore] = None,
        buddy: Optional[Buddy] = None,
        intent_bundle: Optional[dict] = None,
    ):
        self.project_id = project_id
        self.bundle = bundle
        self.llm = llm
        self.store = store or EventStore(project_id)
        self.buddy = buddy or Buddy()
        self.intent_bundle = intent_bundle
        self._heard_any_message = False
        self._prev_text = ""
        self._topic: Optional[tuple[str, str, str]] = None     # (field, value, speaker who proposed it)
        self._since_topic = 0
        self.candidates: dict[str, Candidate] = {}              # field -> best open document suggestion
        self.accepted: list[tuple[Candidate, str]] = []         # (suggestion, accepted by), not saved yet
        self._found_topic: Optional[Candidate] = None           # the suggestion Buddy last asked about
        self._restore_topic()

    # ----------------------------------------------------------- restart
    def _restore_topic(self) -> None:
        """After a restart, use the last saved mention as the topic, for ONE message only."""
        try:
            for event in reversed(self.store.events):
                kind = getattr(event.event_type, "value", event.event_type)
                if kind != "field_updated" or event.actor != EXTRACTOR_ACTOR:
                    continue
                payload = event.payload
                speaker = (payload.get("provenance") or {}).get("speaker")
                if payload.get("field") and payload.get("new_value") and speaker:
                    self._topic = (payload["field"], payload["new_value"], speaker)
                    self._since_topic = TOPIC_LIFETIME - 1      # on_message adds 1, so it counts once
                    return
        except Exception:
            log.warning("Component 1 could not restore the topic after a restart", exc_info=True)

    # ------------------------------------------------------------ intent
    def _intent(self, text: str) -> Optional[tuple[str, float]]:
        if self.intent_bundle is None:
            return None
        try:
            from ..intent.predict import predict_intent
            return predict_intent(self.intent_bundle, text, self._prev_text)
        except Exception:
            log.warning("Component 1 could not classify the intent of a message", exc_info=True)
            return None

    def _endorsement(self, speaker: str, text: str) -> Optional[tuple[str, str]]:
        """(field, value) this message backs up, or None."""
        if self._topic is None or self._since_topic > TOPIC_LIFETIME:
            return None
        field, value, proposer = self._topic
        if speaker == proposer:                      # agreeing with yourself is not confirmation
            return None
        found = self._intent(text)
        if found is None:
            return None
        label, probability = found
        if label in ENDORSING_INTENTS and probability >= MIN_INTENT_CONFIDENCE:
            return field, value
        return None

    # --------------------------------------------------------- documents
    def notice_documents(self) -> None:
        """A document was processed for this team, so Buddy may speak even before any chat."""
        self._heard_any_message = True

    def accept(self, candidate: Candidate, accepted_by: str, via: Optional[str] = None) -> tuple[str, float]:
        """Accept a document suggestion (Blueprint tab or chat). Returns the field's new status."""
        result = accept_candidate(self.store, self.bundle, candidate, accepted_by, via)
        self.accepted.append((candidate, accepted_by))
        if self.candidates.get(candidate.field) is not None and self.candidates[candidate.field].key == candidate.key:
            del self.candidates[candidate.field]
        if self._found_topic is not None and self._found_topic.field == candidate.field:
            self._found_topic = None
        return result

    def _open_found_topic(self, endorse: Optional[tuple[str, str]]) -> Optional[Candidate]:
        """The suggestion this message backs up, if it is the one Buddy asked about and still open."""
        c = self._found_topic
        if endorse is None or c is None or endorse != (c.field, c.value):
            return None
        current = self.candidates.get(c.field)
        return current if current is not None and current.key == c.key else None

    def _after_step(self) -> None:
        """If Buddy just asked about a document suggestion, the next "yes" refers to it."""
        for key in self.buddy.last_step_keys:
            if key[0] == FOUND and key[1] in self.candidates:
                c = self.candidates[key[1]]
                self._found_topic = c
                self._topic = (c.field, c.value, document_speaker(c))
                self._since_topic = 0

    # ------------------------------------------------------------- calls
    def on_message(self, message_id: str, speaker: str, text: str, now: float) -> list[str]:
        """Call for every message written by a PERSON. Never call it for Buddy's own messages."""
        self._heard_any_message = True
        self.buddy.heard_message(now)
        self._since_topic += 1

        endorse = self._endorsement(speaker, text)
        found = self._open_found_topic(endorse)
        result = process_message(self.store, self.bundle, self.llm, message_id, speaker, text,
                                 endorse=None if found else endorse, candidates=self.candidates)
        if found is not None and not result.saved:      # "yes, use that" about Buddy's suggestion
            label, probability = self.accept(found, speaker, via=message_id)
            saved = Signal(SAVED, found.field, (found.value,), detail=_SAVED_DETAIL.get(label, ""),
                           source=f"{found.source_label}, accepted by {speaker}")
            result = MessageResult(saved=[(found.field, found.value)], statuses={found.field: (label, probability)},
                                   signals=[saved] + state_signals(self.store, self.candidates))
            endorse = (found.field, found.value)

        # A message that stated a new value becomes the topic the next "ok" refers to.
        stated = [s for s in result.saved if endorse is None or s != endorse]
        if stated:
            field, value = stated[-1]
            self._topic = (field, value, speaker)
            self._since_topic = 0
        self._prev_text = text
        replies = self.buddy.step(result.signals, now)
        self._after_step()
        return replies

    def tick(self, now: float) -> list[str]:
        """Call regularly (for example every 30 seconds). Lets Buddy ask a question after
        the chat has gone quiet. It never repeats saves and never speaks in an empty chat."""
        if not self._heard_any_message:
            return []
        signals: list[Signal] = [s for s in state_signals(self.store, self.candidates) if s.kind != SAVED]
        replies = self.buddy.step(signals, now)
        self._after_step()
        return replies