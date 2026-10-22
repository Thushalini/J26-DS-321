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

If use_intent=True, the decision-intent model is loaded once and handed to each team's agent,
so a message like "ok, lock that in" counts as one more person agreeing. Off by default.

Documents (needs a database connection):
    process_document(team_id, doc_id, filename, data)   read an upload and store its suggestions
    accept_suggestion(team_id, fact_id, accepted_by)     put a suggestion into the blueprint
    dismiss_suggestion(team_id, fact_id, dismissed_by)   never suggest that value again
Like handle_chat_message, none of them raises.
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
from .documents.candidates import best_candidates, find_group
from .documents.chunk import chunk_pages
from .documents.extract import extract_document
from .documents.parse import DocumentError, parse_document
from .extraction.extractor import LLMFn
from .extraction.gemini_provider import load_env_file
from .extraction.llm import build_extractor_llm
from .intent.predict import MODEL_PATH as INTENT_MODEL_PATH, load_model as load_intent_model
from .state.predict import MODEL_PATH, load_model
from .storage import (
    ensure_c1_tables,
    fail_document,
    finish_document,
    list_agent_messages,
    list_doc_facts,
    load_events,
    mark_processed,
    mirror_old_blueprint,
    save_agent_message,
    save_new_events,
    set_facts_status,
)

READ_FAILED = "Something went wrong while reading this document. Please try uploading it again."

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
        use_intent: bool = False,
    ):
        self._bundle = bundle
        self._llm = llm
        self._config = buddy_config or BuddyConfig()
        self._model_path = model_path
        self._env_file = env_file
        self._conn_factory = conn_factory
        self._use_intent = use_intent
        self._intent_bundle: Optional[dict] = None
        self._intent_tried = False
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

    def _get_intent_bundle(self) -> Optional[dict]:
        """The decision-intent model, loaded once. None means "not used", and nothing else changes."""
        if not self._use_intent or self._intent_tried:
            return self._intent_bundle
        self._intent_tried = True
        try:
            if Path(INTENT_MODEL_PATH).exists():
                self._intent_bundle = load_intent_model(INTENT_MODEL_PATH)
                log.info("Component 1 loaded the decision-intent model")
            else:
                log.warning("Component 1 intent model not found; run: python -m component1.intent.train")
        except Exception:
            log.warning("Component 1 could not load the intent model", exc_info=True)
        return self._intent_bundle

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
                agent = ChatAgent(team_id, self._bundle, self._llm, store=store, buddy=buddy,
                                  intent_bundle=self._get_intent_bundle())
                if conn is not None:
                    self._load_candidates(conn, agent)
                self._agents[team_id] = agent
                self._locks[team_id] = threading.Lock()
                self._seen[team_id] = set()
            return self._agents[team_id], self._locks[team_id], self._seen[team_id]

    @staticmethod
    def _load_candidates(conn, agent: ChatAgent) -> None:
        """Give the agent the team's open document suggestions. A failure only means no suggestions."""
        try:
            agent.candidates = best_candidates(list_doc_facts(conn, agent.project_id))
            if agent.candidates:
                agent.notice_documents()     # open suggestions are worth asking about, even before chat
        except Exception:
            log.warning("Component 1 could not load document suggestions for team %s", agent.project_id,
                        exc_info=True)

    def _persist(self, conn, team_id: str, agent: ChatAgent, replies: list[str],
                 ts_text: str, now: float, after_msg_id: Optional[str]) -> None:
        save_new_events(conn, team_id, agent.store.events)
        while agent.accepted:                  # suggestions accepted in the chat or on the Blueprint tab
            candidate, accepted_by = agent.accepted.pop(0)
            set_facts_status(conn, team_id, candidate.fact_ids, "accepted", decided_by=accepted_by)
        for reply in replies:
            keys = [list(k) for k in agent.buddy.last_step_keys]
            save_agent_message(
                conn, team_id, reply, ts_text, after_msg_id,
                kinds=sorted({k[0] for k in keys}),
                details={"keys": keys, "asked": agent.buddy.last_step_asked, "at": now},
            )
        try:   # last on purpose: Component 2's three-value row must never undo the work above
            changed = mirror_old_blueprint(conn, team_id, agent.store.state)
        except Exception:
            log.warning("Component 1 could not update the blueprints row for team %s", team_id, exc_info=True)
            return
        if not changed:
            return
        try:   # keep Component 2's drift result in step with the row just updated
            if conn.execute("SELECT 1 FROM activities WHERE team_id = ? LIMIT 1", (team_id,)).fetchone():
                from buddy.sync import analyse_team     # called the same way the app's own blueprint save does
                analyse_team(conn, team_id)
                conn.commit()
        except Exception:
            log.warning("Component 1 could not refresh Component 2's drift check for team %s", team_id, exc_info=True)
            rollback = getattr(getattr(conn, "raw", None), "rollback", None)
            if rollback:
                try:
                    rollback()
                except Exception:
                    pass

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
        reply_to: Optional[str] = None,
    ) -> list[str]:
        """Call this after the message is saved. Returns what Buddy said (usually nothing).
        reply_to: the message the person answered with the app's Reply button, if any."""
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
                replies = agent.on_message(message_id, speaker, text, now, reply_to=reply_to)
                if conn is not None:
                    ts_text = timestamp if isinstance(timestamp, str) else _ts_text(now)
                    self._persist(conn, team_id, agent, replies, ts_text, now, message_id)
                return replies
        except Exception:
            log.exception("Component 1 failed on message %s (the chat is not affected)", message_id)
            return []
        finally:
            self._close(conn)

    # -------------------------------------------------------- documents
    def process_document(self, team_id: str, doc_id: str, filename: str, data: bytes) -> dict:
        """Read an uploaded document and store its suggestions. Run it in the background.

        The document row must exist already (storage.create_document). Returns a short summary;
        the same outcome is saved on the row, so the screen can show it. Never raises.
        """
        conn = None
        try:
            conn = self._open()
            if conn is None:
                return {"status": "failed", "error": "Documents need a database connection."}
            try:
                pages = parse_document(filename, data)
            except DocumentError as exc:
                fail_document(conn, doc_id, str(exc))
                return {"status": "failed", "error": str(exc)}
            chunks = chunk_pages(pages)
            ready = self._ensure_ready()               # not ready = rules only, never a failed upload
            fallback = getattr(self._llm, "fallback", None) if ready else None
            facts = extract_document(chunks, fallback_llm=fallback)
            finish_document(conn, doc_id, chunks, facts, pages=len(pages))
            if ready:                                  # let Buddy know about the new suggestions
                agent, lock, _ = self._team(team_id, conn)
                with lock:
                    self._load_candidates(conn, agent)
            return {"status": "done", "chunks": len(chunks), "facts": len(facts)}
        except Exception:
            log.exception("Component 1 could not process document %s", doc_id)
            try:
                if conn is not None:
                    fail_document(conn, doc_id, READ_FAILED)
            except Exception:
                log.warning("Component 1 could not mark document %s as failed", doc_id, exc_info=True)
            return {"status": "failed", "error": READ_FAILED}
        finally:
            self._close(conn)

    def accept_suggestion(self, team_id: str, fact_id: str, accepted_by: str) -> dict:
        """Put a document suggestion into the blueprint. accepted_by is the member's code."""
        conn = None
        try:
            if not self._ensure_ready():
                return {"ok": False, "error": "Component 1 is not ready yet. Please try again shortly."}
            conn = self._open()
            if conn is None:
                return {"ok": False, "error": "Documents need a database connection."}
            agent, lock, _ = self._team(team_id, conn)
            with lock:
                group = find_group(list_doc_facts(conn, team_id), fact_id)
                if group is None:
                    return {"ok": False, "error": "This suggestion has already been accepted or dismissed."}
                label, probability = agent.accept(group, accepted_by)
                now = time.time()
                self._persist(conn, team_id, agent, [], _ts_text(now), now, None)
                self._load_candidates(conn, agent)
                return {"ok": True, "field": group.field, "value": group.value, "status": label}
        except Exception:
            log.exception("Component 1 could not accept suggestion %s", fact_id)
            return {"ok": False, "error": "The suggestion could not be saved. Please try again."}
        finally:
            self._close(conn)

    def dismiss_suggestion(self, team_id: str, fact_id: str, dismissed_by: str) -> dict:
        """Dismiss a suggestion. The same value is never suggested again for that field."""
        conn = None
        try:
            conn = self._open()
            if conn is None:
                return {"ok": False, "error": "Documents need a database connection."}
            with self._registry_lock:
                agent, lock = self._agents.get(team_id), self._locks.get(team_id)
            group = find_group(list_doc_facts(conn, team_id), fact_id)
            if group is None:
                return {"ok": False, "error": "This suggestion has already been accepted or dismissed."}
            set_facts_status(conn, team_id, group.fact_ids, "dismissed", decided_by=dismissed_by)
            if agent is not None:
                with lock:
                    self._load_candidates(conn, agent)
            return {"ok": True, "field": group.field, "value": group.value}
        except Exception:
            log.exception("Component 1 could not dismiss suggestion %s", fact_id)
            return {"ok": False, "error": "The suggestion could not be dismissed. Please try again."}
        finally:
            self._close(conn)

    def refresh_suggestions(self, team_id: str) -> None:
        """Reload a team's suggestions from the database (for example after a document was removed)."""
        conn = None
        try:
            with self._registry_lock:
                agent, lock = self._agents.get(team_id), self._locks.get(team_id)
            if agent is None:
                return                       # loaded fresh when the agent is first needed
            conn = self._open()
            if conn is not None:
                with lock:
                    self._load_candidates(conn, agent)
        except Exception:
            log.warning("Component 1 could not refresh suggestions for team %s", team_id, exc_info=True)
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
    reply_to: Optional[str] = None,
) -> list[str]:
    return default_service().handle_chat_message(team_id, message_id, member_code, text, timestamp, user_id,
                                                 sender_type, reply_to)


def tick_all(now: Timestamp = None) -> dict[str, list[str]]:
    return default_service().tick_all(now)