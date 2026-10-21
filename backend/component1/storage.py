"""Component 1's tables in the shared database (Supabase in production, SQLite for local work).

Works with the app's connection object: conn.execute(sql, params) with ? placeholders. Rows
may be plain tuples (as the app's Postgres connection returns) or named rows. Only plain TEXT and INTEGER columns are used, and JSON
is stored as text, so the same SQL runs on both Postgres and SQLite. Every table starts
with c1_ so nothing clashes with the team's tables, and nothing existing is changed.

    c1_events          the blueprint's full history, one row per change, never edited
    c1_agent_messages  what Buddy said, with room for accept/decline and an audit trail
    c1_processed       which chat messages Component 1 has already handled
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from .common.schema import Blueprint, Event, EventType, FieldName, FieldStatus

TABLES = [
    """CREATE TABLE IF NOT EXISTS c1_events (
        team_id    TEXT NOT NULL,
        version    INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        payload    TEXT NOT NULL,
        actor      TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (team_id, version)
    )""",
    """CREATE TABLE IF NOT EXISTS c1_agent_messages (
        agent_msg_id TEXT PRIMARY KEY,
        team_id      TEXT NOT NULL,
        ts           TEXT NOT NULL,
        after_msg_id TEXT,
        text         TEXT NOT NULL,
        kinds        TEXT NOT NULL DEFAULT '[]',
        details      TEXT NOT NULL DEFAULT '{}',
        status       TEXT NOT NULL DEFAULT 'open',
        responded_by TEXT,
        responded_at TEXT,
        user_value   TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_c1_agent_messages_team_ts ON c1_agent_messages (team_id, ts)",
    """CREATE TABLE IF NOT EXISTS c1_processed (
        team_id      TEXT NOT NULL,
        message_id   TEXT NOT NULL,
        processed_at TEXT NOT NULL,
        PRIMARY KEY (team_id, message_id)
    )""",
]


def _row_dict(cursor: Any, row: Any) -> dict:
    if isinstance(row, dict):
        return dict(row)
    names = [d[0] for d in cursor.description]
    return dict(zip(names, row))


def _all(cursor: Any) -> list[dict]:
    """Every row as a dict, whatever kind of rows the connection returns."""
    return [_row_dict(cursor, r) for r in cursor.fetchall()]


def _one(cursor: Any) -> Optional[dict]:
    row = cursor.fetchone()
    return None if row is None else _row_dict(cursor, row)


def _commit(conn: Any) -> None:
    commit = getattr(conn, "commit", None)
    if commit:
        commit()


def _now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_c1_tables(conn: Any) -> None:
    """Create the tables if they are missing. Safe to run any number of times."""
    for statement in TABLES:
        conn.execute(statement)
    _commit(conn)


# ----------------------------------------------------------------- events
def save_new_events(conn: Any, team_id: str, events: Iterable[Event]) -> int:
    """Save every event not stored yet. Returns how many were added. Nothing is ever updated."""
    row = _one(conn.execute("SELECT MAX(version) AS v FROM c1_events WHERE team_id = ?", (team_id,)))
    last = row["v"] if row is not None and row["v"] is not None else 0
    new = sorted((e for e in events if e.version > last), key=lambda e: e.version)
    for e in new:
        conn.execute(
            "INSERT INTO c1_events (team_id, version, event_type, payload, actor, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (team_id, e.version, e.event_type.value, json.dumps(e.payload), e.actor, e.created_at.isoformat()),
        )
    _commit(conn)
    return len(new)


def load_events(conn: Any, team_id: str) -> list[Event]:
    rows = _all(conn.execute(
        "SELECT version, event_type, payload, actor, created_at FROM c1_events WHERE team_id = ? ORDER BY version",
        (team_id,),
    ))
    return [
        Event(
            project_id=team_id,
            version=r["version"],
            event_type=EventType(r["event_type"]),
            payload=json.loads(r["payload"]),
            actor=r["actor"],
            created_at=datetime.fromisoformat(r["created_at"]),
        )
        for r in rows
    ]


# ------------------------------------------------------ processed messages
def mark_processed(conn: Any, team_id: str, message_id: str) -> bool:
    """Record that a message was handled. Returns False if it had been handled already."""
    seen = _one(conn.execute(
        "SELECT 1 AS x FROM c1_processed WHERE team_id = ? AND message_id = ?", (team_id, message_id)
    ))
    if seen is not None:
        return False
    conn.execute(
        "INSERT INTO c1_processed (team_id, message_id, processed_at) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
        (team_id, message_id, _now_text()),
    )
    _commit(conn)
    return True


# --------------------------------------------------------- Buddy's messages
def save_agent_message(
    conn: Any,
    team_id: str,
    text: str,
    ts: str,
    after_msg_id: Optional[str] = None,
    kinds: Iterable[str] = (),
    details: Optional[dict] = None,
) -> str:
    """Store one thing Buddy said. details holds the reasons behind it (for the audit trail)."""
    agent_msg_id = "b_" + uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO c1_agent_messages (agent_msg_id, team_id, ts, after_msg_id, text, kinds, details) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (agent_msg_id, team_id, ts, after_msg_id, text, json.dumps(list(kinds)), json.dumps(details or {})),
    )
    _commit(conn)
    return agent_msg_id


def list_agent_messages(conn: Any, team_id: str) -> list[dict]:
    """All of Buddy's messages for a team, oldest first."""
    rows = _all(conn.execute(
        "SELECT agent_msg_id, ts, after_msg_id, text, kinds, details, status FROM c1_agent_messages "
        "WHERE team_id = ? ORDER BY ts, agent_msg_id",
        (team_id,),
    ))
    return [_assistant_item(r) for r in rows]


def _assistant_item(r: Any) -> dict:
    return {
        "id": r["agent_msg_id"],
        "sender_type": "assistant",
        "sender_name": "Buddy",
        "ts": r["ts"],
        "text": r["text"],
        "after_msg_id": r["after_msg_id"],
        "kinds": json.loads(r["kinds"] or "[]"),
        "details": json.loads(r["details"] or "{}"),
        "status": r["status"],
    }


# ------------------------------------------------------ the one timeline
def timeline(
    conn: Any,
    team_id: str,
    limit: int = 50,
    before_ts: Optional[str] = None,
    after_ts: Optional[str] = None,
) -> list[dict]:
    """Members' messages and Buddy's replies as ONE list, oldest first.

    Returns the latest `limit` items. before_ts pages back to older ones; after_ts gives only
    newer ones (for the screen's refresh). At the same second a member's message comes first.

    ASSUMES the shared tables: chat_messages(msg_id, team_id, user_id, ts, text, reply_to),
    users(user_id, name) and team_members(team_id, user_id, member_code, team_role).
    """
    filters, params = "", []
    if after_ts:
        filters += " AND {ts} > ?"
        params.append(after_ts)
    if before_ts:
        filters += " AND {ts} < ?"
        params.append(before_ts)

    member_rows = _all(conn.execute(
        "SELECT c.msg_id AS id, c.ts AS ts, c.text AS text, c.reply_to AS reply_to, "
        "c.user_id AS user_id, u.name AS sender_name, m.member_code AS member_code, m.team_role AS team_role "
        "FROM chat_messages c "
        "LEFT JOIN users u ON u.user_id = c.user_id "
        "LEFT JOIN team_members m ON m.team_id = c.team_id AND m.user_id = c.user_id "
        "WHERE c.team_id = ?" + filters.format(ts="c.ts") + " ORDER BY c.ts DESC, c.msg_id DESC LIMIT ?",
        [team_id, *params, limit],
    ))
    agent_rows = _all(conn.execute(
        "SELECT agent_msg_id, ts, after_msg_id, text, kinds, details, status FROM c1_agent_messages "
        "WHERE team_id = ?" + filters.format(ts="ts") + " ORDER BY ts DESC, agent_msg_id DESC LIMIT ?",
        [team_id, *params, limit],
    ))

    items = [
        {
            "id": r["id"], "sender_type": "member", "sender_name": r["sender_name"], "user_id": r["user_id"],
            "member_code": r["member_code"], "team_role": r["team_role"],
            "ts": r["ts"], "text": r["text"], "reply_to": r["reply_to"],
        }
        for r in member_rows
    ] + [_assistant_item(r) for r in agent_rows]
    items.sort(key=lambda i: (i["ts"], 0 if i["sender_type"] == "member" else 1, i["id"]))
    return items[-limit:]


# ------------------------------------- the existing chat listing, plus Buddy
def merge_agent_messages(conn: Any, team_id: str, messages: list[dict], limit: int = 200) -> list[dict]:
    """Add Buddy's messages to the list the existing chat endpoint already built.

    `messages` is that endpoint's list, oldest first, with keys msg_id, sender, member_code, ts,
    text, reply_to, mine and team_role. Buddy's messages get the same keys, plus sender_type
    "assistant", so the screen can style them. Members' messages get sender_type "member".
    """
    members = [dict(m, sender_type=m.get("sender_type", "member")) for m in messages]
    sql = (
        "SELECT agent_msg_id, ts, after_msg_id, text, kinds, details, status "
        "FROM c1_agent_messages WHERE team_id = ?"
    )
    params: list = [team_id]
    if members and len(members) >= limit:          # older Buddy messages would be outside the window
        sql += " AND ts >= ?"
        params.append(members[0]["ts"])
    sql += " ORDER BY ts DESC, agent_msg_id DESC LIMIT ?"
    params.append(limit)
    buddy = [
        {
            "msg_id": r["agent_msg_id"], "sender": "Buddy", "member_code": None, "ts": r["ts"],
            "text": r["text"], "reply_to": r["after_msg_id"], "mine": False, "team_role": "assistant",
            "sender_type": "assistant", "kinds": json.loads(r["kinds"] or "[]"), "status": r["status"],
        }
        for r in _all(conn.execute(sql, params))
    ]
    merged = members + buddy
    merged.sort(key=lambda m: (m["ts"], 0 if m["sender_type"] == "member" else 1, m["msg_id"]))
    return merged[-limit:]


# ----------------------------- keep Component 2's three-value blueprint current
MIRROR = {
    "target": FieldName.TARGET_VARIABLE,
    "algorithm": FieldName.ALGORITHM,
    "metric": FieldName.EVALUATION_METRIC,
}


def mirror_old_blueprint(conn: Any, team_id: str, state: Blueprint, updated_by: str = "component1") -> bool:
    """Copy target / algorithm / metric into the app's `blueprints` row, so Component 2's drift
    check keeps working. Only values that are KNOWN or CONFIRMED are copied, and a value that
    is not settled never erases what is already there (a person may have typed it by hand).
    Returns False if there was nothing to copy.
    """
    settled = (FieldStatus.KNOWN, FieldStatus.CONFIRMED)
    values = {
        column: state.fields[name].value
        for column, name in MIRROR.items()
        if state.fields[name].status in settled
    }
    if not values:
        return False
    conn.execute(
        "INSERT INTO blueprints (team_id, target, algorithm, metric, updated_by, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (team_id) DO UPDATE SET "
        "target = COALESCE(excluded.target, blueprints.target), "
        "algorithm = COALESCE(excluded.algorithm, blueprints.algorithm), "
        "metric = COALESCE(excluded.metric, blueprints.metric), "
        "updated_by = excluded.updated_by, updated_at = excluded.updated_at",
        (team_id, values.get("target"), values.get("algorithm"), values.get("metric"), updated_by, _now_text()),
    )
    _commit(conn)
    return True