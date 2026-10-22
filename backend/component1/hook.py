"""The few lines that connect Component 1 to the shared app (buddy/router.py and main.py).

    router.py, post_chat, after the message is saved:
        run_in_background(background_tasks, team_id, msg_id, member_code, user["user_id"], ts, text)

    router.py, get_chat, real-team branch, after the existing `messages` list is built:
        messages = merge_chat(conn, team_id, messages, limit)

    main.py, once at startup:
        start_timer()

Everything here is safe: none of it can raise into the chat, and each helper falls back to
doing nothing if Component 1 has a problem.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from .service import AgentService
from .storage import ensure_c1_tables, merge_agent_messages

log = logging.getLogger("component1")

_service: Optional[AgentService] = None
_service_lock = threading.Lock()
_tables_ready = False


def service() -> AgentService:
    """The one shared service. It uses the app's own connect(), so it follows DATABASE_URL.
    The decision-intent model is switched on here, so "ok, lock that in" counts as agreement."""
    global _service
    with _service_lock:
        if _service is None:
            from component2.db import connect
            _service = AgentService(conn_factory=connect, use_intent=True)
        return _service


def _run(team_id, msg_id, member_code, user_id, ts, text, reply_to=None) -> None:
    service().handle_chat_message(team_id, msg_id, member_code, text, ts, user_id=user_id, reply_to=reply_to)


def run_in_background(background_tasks: Any, team_id: str, msg_id: str, member_code: Optional[str],
                      user_id: Optional[str], ts: str, text: str, reply_to: Optional[str] = None) -> None:
    """Hand a saved message to Component 1 WITHOUT making the sender wait. Never raises.
    reply_to: the message the person answered with the Reply button, if any (helps reply linking)."""
    try:
        background_tasks.add_task(_run, team_id, msg_id, member_code, user_id, ts, text, reply_to)
    except Exception:
        log.exception("Component 1 could not be scheduled for message %s (the chat is not affected)", msg_id)


def _rollback(conn: Any) -> None:
    rollback = getattr(getattr(conn, "raw", None), "rollback", None)
    if rollback:
        try:
            rollback()
        except Exception:
            pass


def merge_chat(conn: Any, team_id: str, messages: list[dict], limit: int = 200) -> list[dict]:
    """Add Buddy's messages to the chat listing. On any problem, return the list unchanged."""
    global _tables_ready
    try:
        if not _tables_ready:
            ensure_c1_tables(conn)       # the first listing can come before the first message
            _tables_ready = True
        return merge_agent_messages(conn, team_id, messages, limit)
    except Exception:
        log.exception("Component 1 could not add Buddy's messages to the listing (showing members only)")
        _rollback(conn)
        return messages


class Ticker:
    """Calls a function every few seconds on a background thread."""

    def __init__(self, func: Callable[[], Any], interval_seconds: float):
        self._func = func
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._func()
            except Exception:
                log.exception("Component 1 timer step failed")

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="component1-ticker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_ticker: Optional[Ticker] = None


def start_timer(interval_seconds: float = 30.0) -> None:
    """Start the timer that lets Buddy speak after a quiet spell. Safe to call twice.

    With several server workers each would run its own timer, so run ONE worker for now."""
    global _ticker
    if _ticker is None:
        _ticker = Ticker(lambda: service().tick_all(), interval_seconds)
    _ticker.start()


def stop_timer() -> None:
    if _ticker is not None:
        _ticker.stop()