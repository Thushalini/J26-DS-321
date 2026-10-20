"""One team's assistant: hears each chat message and decides what, if anything, to say.

It holds the blueprint (an EventStore) and Buddy's speaking rules for ONE team. It
knows nothing about databases, web servers or the chat screen, so it can be tested with
plain Python. Later a thin layer rebuilds it from saved events and posts its replies.

    agent = ChatAgent("team_1", bundle, llm)
    replies = agent.on_message("M1", "Member_A", "I think we should predict arrivals", now=0)
    replies = agent.tick(now=300)    # call on a timer, so Buddy can speak after a quiet spell
"""
from __future__ import annotations

from typing import Optional

from ..common.events import EventStore
from ..extraction.extractor import LLMFn
from ..pipeline import process_message, state_signals
from .policy import Buddy
from .signals import SAVED, Signal


class ChatAgent:
    def __init__(
        self,
        project_id: str,
        bundle: dict,
        llm: LLMFn,
        store: Optional[EventStore] = None,
        buddy: Optional[Buddy] = None,
    ):
        self.project_id = project_id
        self.bundle = bundle
        self.llm = llm
        self.store = store or EventStore(project_id)
        self.buddy = buddy or Buddy()
        self._heard_any_message = False

    def on_message(self, message_id: str, speaker: str, text: str, now: float) -> list[str]:
        """Call for every message written by a PERSON. Never call it for Buddy's own messages."""
        self._heard_any_message = True
        self.buddy.heard_message(now)
        result = process_message(self.store, self.bundle, self.llm, message_id, speaker, text)
        return self.buddy.step(result.signals, now)

    def tick(self, now: float) -> list[str]:
        """Call regularly (for example every 30 seconds). Lets Buddy ask a question after
        the chat has gone quiet. It never repeats saves and never speaks in an empty chat."""
        if not self._heard_any_message:
            return []
        signals: list[Signal] = [s for s in state_signals(self.store) if s.kind != SAVED]
        return self.buddy.step(signals, now)