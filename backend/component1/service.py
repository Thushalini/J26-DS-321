"""The one function the chat endpoint calls after it has saved a message.

    handle_chat_message(team_id, message_id, member_code, text, timestamp)

It NEVER raises, so the chat can never break because of Component 1. It returns what Buddy
said (usually nothing). When a database connection is configured, it also:
  - saves every blueprint change in c1_events,
  - saves Buddy's replies in c1_agent_messages (the chat screen reads them with storage.timeline),
  - remembers which messages it has handled, so a restart never repeats work,
  - rebuilds a team's blueprint and Buddy's memory from the saved rows after a restart.

Run it in the background so a slow language model never delays the sender (for example
with FastAPI BackgroundTasks). Without a connection it keeps everything in memory (tests).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from calendar import timegm
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .agent.chat_agent import ChatAgent
from .agent.policy import Buddy, BuddyConfig
from .common.events import EventStore
from .extraction.extractor import LLMFn
from .extraction.gemini_provider import load_env_file
from .extraction.llm import build_extractor_llm
from .state.predict import MODEL_PATH, load_model
from .storage import (
    ensure_c1_tables,
    list_agent_messages,
    load_events,
    mark_processed,
    mirror_old_blueprint,
    save_agent_message,
    save_new_events,
)

log = logging.getLogger("component1")

BACKEND_DIR = Path(__file__).resolve().parents[1]
Timestamp = Union[None, int, float, str, datetime]
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


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
        return float(timegm(time.strptime(text, TS_FORMAT)))
    except ValueError:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _seconds_or_now(ts: Timestamp) -> float:
    """A bad timestamp must not lose the message. Fall back to the current time."""
    try:
        return to_seconds(ts)
    except Exception:
        log.warning("Component 1 could not read the timestamp %r; using the current time", ts)
        return time.time()


def _ts_text(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime(TS_FORMAT)


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
        conn_factory: Optional[Callable[[], Any]] = None,
    ):
        self._bundle = bundle
        self._llm = llm
        self._config = buddy_config or BuddyConfig()
        self._model_path = model_path
        self._env_file = env_file
        self._conn_factory = conn_factory
        self._tables_ready = False
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
                # Rules plus Groq or Gemini (C1_LLM_PROVIDER). Without a key: rules only.
                self._llm, how = build_extractor_llm()
                log.info("Component 1 extracts with: %s", how)
            return True
        except Exception as exc:  # not ready is a normal state, never an error for the chat
            log.warning("Component 1 is not ready: %s", exc)
            return False

    def _open(self):
        if self._conn_factory is None:
            return None
        conn = self._conn_factory()
        if not self._tables_ready:
            ensure_c1_tables(conn)
            self._tables_ready = True
        return conn

    @staticmethod
    def _close(conn) -> None:
        close = getattr(conn, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass

    def _team(self, team_id: str, conn):
        with self._registry_lock:
            if team_id not in self._agents:
                store, buddy = None, Buddy(self._config)
                if conn is not None:                       # rebuild from what was saved before
                    store = EventStore.from_events(team_id, load_events(conn, team_id))
                    for saved in list_agent_messages(conn, team_id):
                        d = saved["details"]
                        buddy.restore(d.get("keys", []), d.get("at", 0.0), d.get("asked", False))
                self._agents[team_id] = ChatAgent(team_id, self._bundle, self._llm, store=store, buddy=buddy)
                self._locks[team_id] = threading.Lock()
                self._seen[team_id] = set()
            return self._agents[team_id], self._locks[team_id], self._seen[team_id]

    def _persist(self, conn, team_id: str, agent: ChatAgent, replies: list[str],
                 ts_text: str, now: float, after_msg_id: Optional[str]) -> None:
        save_new_events(conn, team_id, agent.store.events)
        for reply in replies:
            keys = [list(k) for k in agent.buddy.last_step_keys]
            save_agent_message(
                conn, team_id, reply, ts_text, after_msg_id,
                kinds=sorted({k[0] for k in keys}),
                details={"keys": keys, "asked": agent.buddy.last_step_asked, "at": now},
            )
        try:   # last on purpose: Component 2's three-value row must never undo the work above
            mirror_old_blueprint(conn, team_id, agent.store.state)
        except Exception:
            log.warning("Component 1 could not update the blueprints row for team %s", team_id, exc_info=True)

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
        """Call this after the message is saved. Returns what Buddy said (usually nothing)."""
        conn = None
        try:
            if sender_type != "member":      # never react to Buddy's own messages
                return []
            if not text or not text.strip():
                return []
            if not self._ensure_ready():
                return []
            conn = self._open()
            agent, lock, seen = self._team(team_id, conn)
            with lock:                        # one message at a time per team
                if conn is not None:
                    if not mark_processed(conn, team_id, message_id):   # never handle a message twice
                        return []
                elif message_id in seen:
                    return []
                else:
                    seen.add(message_id)      # a failed message is skipped, not retried
                speaker = speaker_for(member_code, user_id)
                now = _seconds_or_now(timestamp)
                replies = agent.on_message(message_id, speaker, text, now)
                if conn is not None:
                    ts_text = timestamp if isinstance(timestamp, str) else _ts_text(now)
                    self._persist(conn, team_id, agent, replies, ts_text, now, message_id)
                return replies
        except Exception:
            log.exception("Component 1 failed on message %s (the chat is not affected)", message_id)
            return []
        finally:
            self._close(conn)

    def tick(self, team_id: str, now: Timestamp = None) -> list[str]:
        """Call on a timer so Buddy can speak after a quiet spell."""
        conn = None
        try:
            with self._registry_lock:
                agent, lock = self._agents.get(team_id), self._locks.get(team_id)
            if agent is None:
                return []
            with lock:
                seconds = _seconds_or_now(now)
                replies = agent.tick(seconds)
                if replies:
                    conn = self._open()
                    if conn is not None:
                        self._persist(conn, team_id, agent, replies, _ts_text(seconds), seconds, None)
                return replies
        except Exception:
            log.exception("Component 1 tick failed for team %s (the chat is not affected)", team_id)
            return []
        finally:
            self._close(conn)

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


def configure(**kwargs) -> AgentService:
    """Set up the shared service once, when the app starts. For example:

        configure(conn_factory=component2.db.connect)
    """
    global _default
    _default = AgentService(**kwargs)
    return _default


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