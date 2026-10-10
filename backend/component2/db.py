import json
import sqlite3

from component2.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS activities (
    activity_id TEXT PRIMARY KEY,
    team_id     TEXT NOT NULL,
    source      TEXT NOT NULL,
    member_id   TEXT,
    ts          TEXT NOT NULL,
    text        TEXT NOT NULL,
    meta        TEXT
);
"""


def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    return conn


def save_activities(conn, rows):
    before = conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
    for r in rows:
        conn.execute(
            "INSERT OR IGNORE INTO activities VALUES (?, ?, ?, ?, ?, ?, ?)",
            (r["activity_id"], r["team_id"], r["source"], r["member_id"],
             r["ts"], r["text"], json.dumps(r["meta"])),
        )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
    return after - before