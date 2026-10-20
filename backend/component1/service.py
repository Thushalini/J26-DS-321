"""The one function the chat endpoint calls after it has saved a message.

    replies = handle_chat_message(team_id, message_id, member_code, text, timestamp)

It NEVER raises, so the chat can never break because of Component 1. It returns an empty
list, or the text of what Buddy wants to say; the caller saves each reply as an assistant
message. Run it in the background so a slow language model never delays the sender (for
example with FastAPI BackgroundTasks).

State is kept in memory for now: after a server restart a team's agent starts empty.
Rebuilding it from saved events is a later step.
"""
from __future__ import annotations

import logging
import threading
import time
from calendar import timegm
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .agent.chat_agent import ChatAgent
from .agent.policy import Buddy, BuddyConfig
from .extraction.extractor import LLMFn
from .extraction.gemini_provider import load_env_file, make_gemini_llm
from .state.predict import MODEL_PATH, load_model

log = logging.getLogger("component1")

BACKEND_DIR = Path(__file__).resolve().parents[1]
Timestamp = Union[None, int, float, str, datetime]


def to_seconds(ts: Timestamp) -> float:
    """Turn the app's text timestamp (2026-10-19T10:21:07Z), a datetime or a number into seconds."""
    if ts is None:
        return time.time()
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime):
        return ts.timestamp()
    text = str(ts).strip()
    try:
        return float(timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _seconds_or_now(ts: Timestamp) -> float:
    """A bad timestamp must not lose the message. Fall back to the current time."""
    try:
        return to_seconds(ts)
    except Exception:
        log.warning("Component 1 could not read the timestamp %r; using the current time", ts)
        return time.time()


def speaker_for(member_code: Optional[str], user_id: Optional[str] = None) -> str:
    """The anonymised speaker: member code, else the user id, else "unknown".

    If many messages end up as "unknown", the state classifier cannot tell speakers apart,
    so nothing can ever become confirmed. Make sure senders have a member code.
    """
    code = (member_code or "").strip()
    if code:
        return code
    return f"user:{user_id}" if user_id else "unknown"


class AgentService:
    """Keeps one ChatAgent per team and hands each person's message to it."""

    def __init__(
        self,
        bundle: Optional[dict] = None,
        llm: Optional[LLMFn] = None,
        buddy_config: Optional[BuddyConfig] = None,
        model_path: Path = MODEL_PATH,
        env_file: Optional[Path] = BACKEND_DIR / ".env",
    ):
        self._bundle = bundle
        self._llm = llm
        self._config = buddy_config or BuddyConfig()
        self._model_path = model_path
        self._env_file = env_file
        self._agents: dict[str, ChatAgent] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._seen: dict[str, set[str]] = {}
        self._registry_lock = threading.Lock()

    # ----------------------------------------------------------- set-up
    def _ensure_ready(self) -> bool:
        if self._bundle is not None and self._llm is not None:
            return True
        try:
            if self._bundle is None:
                if not Path(self._model_path).exists():
                    raise FileNotFoundError("The state model is missing. Run: python -m component1.state.train")
                self._bundle = load_model(self._model_path)
            if self._llm is None:
                if self._env_file is not None:
                    load_env_file(self._env_file)
                self._llm = make_gemini_llm()
            return True
        except Exception as exc:  # not ready is a normal state, never an error for the chat
            log.warning("Component 1 is not ready: %s", exc)
            return False

    def _team(self, team_id: str):
        with self._registry_lock:
            if team_id not in self._agents:
                self._agents[team_id] = ChatAgent(team_id, self._bundle, self._llm, buddy=Buddy(self._config))
                self._locks[team_id] = threading.Lock()
                self._seen[team_id] = set()
            return self._agents[team_id], self._locks[team_id], self._seen[team_id]

    # ------------------------------------------------------------ calls
    def handle_chat_message(
        self,
        team_id: str,
        message_id: str,
        member_code: Optional[str],
        text: str,
        timestamp: Timestamp = None,
        user_id: Optional[str] = None,
        sender_type: str = "member",
    ) -> list[str]:
        """Call this after the message is saved. Returns what Buddy wants to say (usually nothing)."""
        try:
            if sender_type != "member":      # never react to Buddy's own messages
                return []
            if not text or not text.strip():
                return []
            if not self._ensure_ready():
                return []
            agent, lock, seen = self._team(team_id)
            with lock:                        # one message at a time per team
                if message_id in seen:        # never process the same message twice
                    return []
                seen.add(message_id)          # a failed message is skipped, not retried
                speaker = speaker_for(member_code, user_id)
                return agent.on_message(message_id, speaker, text, _seconds_or_now(timestamp))
        except Exception:
            log.exception("Component 1 failed on message %s (the chat is not affected)", message_id)
            return []

    def tick(self, team_id: str, now: Timestamp = None) -> list[str]:
        """Call on a timer so Buddy can speak after a quiet spell."""
        try:
            with self._registry_lock:
                agent, lock = self._agents.get(team_id), self._locks.get(team_id)
            if agent is None:
                return []
            with lock:
                return agent.tick(_seconds_or_now(now))
        except Exception:
            log.exception("Component 1 tick failed for team %s (the chat is not affected)", team_id)
            return []

    def tick_all(self, now: Timestamp = None) -> dict[str, list[str]]:
        """Tick every team that has an agent. Returns only the teams that have something to say."""
        with self._registry_lock:
            teams = list(self._agents)
        out: dict[str, list[str]] = {}
        for team_id in teams:
            replies = self.tick(team_id, now)
            if replies:
                out[team_id] = replies
        return out


_default: Optional[AgentService] = None


def default_service() -> AgentService:
    global _default
    if _default is None:
        _default = AgentService()
    return _default


def handle_chat_message(
    team_id: str,
    message_id: str,
    member_code: Optional[str],
    text: str,
    timestamp: Timestamp = None,
    user_id: Optional[str] = None,
    sender_type: str = "member",
) -> list[str]:
    return default_service().handle_chat_message(team_id, message_id, member_code, text, timestamp, user_id, sender_type)


def tick_all(now: Timestamp = None) -> dict[str, list[str]]:
    return default_service().tick_all(now)