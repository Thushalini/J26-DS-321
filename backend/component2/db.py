"""
Database layer. Works with BOTH:
  - PostgreSQL  if DATABASE_URL is set in backend/.env   (the proposal's database)
  - SQLite      otherwise (a single file, handy for quick tests)

Every other file only uses connect(), save_activities(), save_labels(),
save_drift() and conn.execute(...), so they never need to know which one is used.
"""
import json
import sqlite3

from component2.config import DATABASE_URL, DB_PATH

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

CREATE TABLE IF NOT EXISTS stage_labels (
    activity_id TEXT NOT NULL,
    labeller    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    confidence  REAL,
    PRIMARY KEY (activity_id, labeller)
);

CREATE TABLE IF NOT EXISTS drift_results (
    team_id     TEXT NOT NULL,
    window_end  TEXT NOT NULL,
    dimension   TEXT NOT NULL,
    planned     TEXT,
    seen_top    TEXT,
    drift_score REAL,
    severity    TEXT NOT NULL,
    explanation TEXT,
    PRIMARY KEY (team_id, window_end, dimension)
);

CREATE INDEX IF NOT EXISTS ix_activities_team_ts ON activities (team_id, ts);

CREATE TABLE IF NOT EXISTS role_trends (
    team_id        TEXT NOT NULL,
    member_id      TEXT NOT NULL,
    week           TEXT NOT NULL,
    stage_mix      TEXT,
    chat           TEXT,
    dominant_stage TEXT,
    PRIMARY KEY (team_id, member_id, week)
);

"""

_warned_sqlite = False          # print the "using SQLite" note only once per run


class Connection:
    """Gives SQLite and PostgreSQL the same methods, so no other file needs to care."""

    def __init__(self, url=DATABASE_URL):
        url = url.strip()
        self.is_postgres = url.startswith(("postgresql://", "postgres://"))
        if url and not self.is_postgres:
            # A typo here used to fall back to the old local SQLite file without any warning
            raise ValueError("DATABASE_URL in backend/.env must start with postgresql:// or postgres:// "
                             "(leave it empty to use the local SQLite file).")
        if self.is_postgres:
            import psycopg                      # pip install "psycopg[binary]"
            self.raw = psycopg.connect(url, connect_timeout=50)
        else:
            global _warned_sqlite
            if not _warned_sqlite:
                print(f"NOTE: DATABASE_URL is empty - using local SQLite file {DB_PATH}")
                _warned_sqlite = True
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            # FastAPI may run one request's dependencies on different worker threads; the connection
            # is still used by one request at a time, so sharing it across threads is safe
            self.raw = sqlite3.connect(DB_PATH, check_same_thread=False)

    def _sql(self, sql):
        # SQLite writes placeholders as ?, PostgreSQL (psycopg) as %s
        return sql.replace("?", "%s") if self.is_postgres else sql

    def execute(self, sql, params=()):
        cur = self.raw.cursor()
        cur.execute(self._sql(sql), params)
        return cur

    def executemany(self, sql, rows):
        cur = self.raw.cursor()
        cur.executemany(self._sql(sql), rows)
        return cur

    def commit(self):
        self.raw.commit()

    def close(self):
        self.raw.close()


def connect(create_tables=True):
    """create_tables=False skips the CREATE TABLE step (faster for the API, tables already exist)."""
    conn = Connection()
    if create_tables:
        for statement in SCHEMA.split(";"):
            if statement.strip():
                conn.execute(statement)
        conn.commit()
    return conn


def count_rows(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def save_activities(conn, rows):
    """Adds new activities; ones already saved are updated (e.g. re-dated notebook cells).
    Returns how many were new."""
    before = count_rows(conn, "activities")
    conn.executemany(
        "INSERT INTO activities VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (activity_id) DO UPDATE "
        "SET member_id = excluded.member_id, ts = excluded.ts, "
        "text = excluded.text, meta = excluded.meta",
        [(r["activity_id"], r["team_id"], r["source"], r["member_id"],
          r["ts"], r["text"], json.dumps(r["meta"])) for r in rows],
    )
    conn.commit()
    return count_rows(conn, "activities") - before


def save_labels(conn, labeller, labels):
    """labels: list of (activity_id, stage, confidence). Re-running replaces old labels."""
    conn.executemany(
        "INSERT INTO stage_labels VALUES (?, ?, ?, ?) "
        "ON CONFLICT (activity_id, labeller) DO UPDATE "
        "SET stage = excluded.stage, confidence = excluded.confidence",
        [(aid, labeller, str(stage), None if conf is None else float(conf)) for aid, stage, conf in labels],
    )
    conn.commit()


def save_drift(conn, results):
    """results: list of dicts from detect_drift. Re-running replaces old results."""
    conn.executemany(
        "INSERT INTO drift_results VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (team_id, window_end, dimension) DO UPDATE "
        "SET planned = excluded.planned, seen_top = excluded.seen_top, "
        "drift_score = excluded.drift_score, severity = excluded.severity, "
        "explanation = excluded.explanation",
        [(r["team_id"], r["window_end"], r["dimension"], r["planned"], r["seen_top"],
          r["drift_score"], r["severity"], r["explanation"]) for r in results],
    )
    conn.commit()

def save_roles(conn, rows):
    """rows: list of dicts from role_tracker. Re-running replaces old results."""
    conn.executemany(
        "INSERT INTO role_trends VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (team_id, member_id, week) DO UPDATE "
        "SET stage_mix = excluded.stage_mix, chat = excluded.chat, dominant_stage = excluded.dominant_stage",
        [(r["team_id"], r["member_id"], r["week"], json.dumps(r["stage_mix"]), json.dumps(r["chat"]),
          r["dominant_stage"]) for r in rows],
    )
    conn.commit()