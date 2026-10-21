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
"""
from __future__ import annotations

import logging
from typing import Optional

from ..common.events import EventStore
from ..extraction.extractor import LLMFn
from ..pipeline import process_message, state_signals
from ..state.features import EXTRACTOR_ACTOR
from .policy import Buddy
from .signals import SAVED, Signal

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

    # ------------------------------------------------------------- calls
    def on_message(self, message_id: str, speaker: str, text: str, now: float) -> list[str]:
        """Call for every message written by a PERSON. Never call it for Buddy's own messages."""
        self._heard_any_message = True
        self.buddy.heard_message(now)
        self._since_topic += 1

        endorse = self._endorsement(speaker, text)
        result = process_message(self.store, self.bundle, self.llm, message_id, speaker, text, endorse=endorse)

        # A message that stated a new value becomes the topic the next "ok" refers to.
        stated = [s for s in result.saved if endorse is None or s != endorse]
        if stated:
            field, value = stated[-1]
            self._topic = (field, value, speaker)
            self._since_topic = 0
        self._prev_text = text
        return self.buddy.step(result.signals, now)

    def tick(self, now: float) -> list[str]:
        """Call regularly (for example every 30 seconds). Lets Buddy ask a question after
        the chat has gone quiet. It never repeats saves and never speaks in an empty chat."""
        if not self._heard_any_message:
            return []
        signals: list[Signal] = [s for s in state_signals(self.store) if s.kind != SAVED]
        return self.buddy.step(signals, now)