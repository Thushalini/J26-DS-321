"""App tables (added next to Component 2's tables in the same database).

Plain SQL that works on both PostgreSQL (Supabase) and SQLite. IDs are random
text, so no SERIAL / AUTOINCREMENT differences between the two.
"""
import secrets
from datetime import datetime, timedelta, timezone

from component2.db import connect

APP_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    role          TEXT NOT NULL,               -- student | supervisor | examiner
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS password_resets (
    user_id    TEXT PRIMARY KEY,               -- one live reset code per user (a new request replaces it)
    code_hash  TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS teams (
    team_id      TEXT PRIMARY KEY,             -- research teams keep their old id, e.g. RMMcMurray__Math-4810_Capstone
    name         TEXT NOT NULL,
    repo_url     TEXT,
    is_research  INTEGER NOT NULL DEFAULT 0,   -- 1 = Phase 1 research dataset, 0 = real team
    join_code    TEXT UNIQUE,
    created_by   TEXT,
    created_at   TEXT NOT NULL,
    sync_status  TEXT,                         -- idle | running | done | error
    sync_message TEXT,
    sync_started TEXT,
    last_synced  TEXT,
    unlinked_authors TEXT                      -- JSON: git authors not linked to a member at the last sync
);

CREATE TABLE IF NOT EXISTS team_members (
    team_id        TEXT NOT NULL,
    user_id        TEXT NOT NULL,
    team_role      TEXT NOT NULL,              -- lead | member | supervisor
    member_code    TEXT,                       -- the M-xxxxxx code used in activities (NULL for supervisors)
    git_identities TEXT,                       -- JSON list of names / emails used in this member's commits
    joined_at      TEXT NOT NULL,
    PRIMARY KEY (team_id, user_id)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    msg_id   TEXT PRIMARY KEY,
    team_id  TEXT NOT NULL,
    user_id  TEXT NOT NULL,
    ts       TEXT NOT NULL,
    text     TEXT NOT NULL,
    reply_to TEXT
);

CREATE INDEX IF NOT EXISTS ix_chat_team_ts ON chat_messages (team_id, ts);

CREATE TABLE IF NOT EXISTS blueprints (
    team_id    TEXT PRIMARY KEY,
    target     TEXT,
    algorithm  TEXT,
    metric     TEXT,
    notes      TEXT,
    updated_by TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    team_id     TEXT,
    page        TEXT,
    rating      INTEGER,
    comment     TEXT,
    ts          TEXT NOT NULL
);
"""

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"          # same format as Component 2's activities.ts


def now(offset=None):
    t = datetime.now(timezone.utc)
    if offset:
        t += offset
    return t.strftime(TS_FORMAT)


def new_id(prefix):
    return f"{prefix}_{secrets.token_hex(8)}"


def ensure_tables(conn):
    for statement in APP_SCHEMA.split(";"):
        if statement.strip():
            conn.execute(statement)
    conn.commit()


_ready = False


def open_conn():
    """Connection for one request. The first call also creates any missing tables (C2's and the app's)."""
    global _ready
    conn = connect(create_tables=not _ready)
    if not _ready:
        ensure_tables(conn)
        _ready = True
    return conn


def rows_as_dicts(cur):
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def one_as_dict(cur):
    rows = rows_as_dicts(cur)
    return rows[0] if rows else None


SESSION_DAYS = 14


def session_expiry():
    return now(timedelta(days=SESSION_DAYS))
