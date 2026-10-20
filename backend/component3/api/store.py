"""Component 3's alerts in the team database.

Uses Component 2's Connection wrapper, so it works on PostgreSQL (DATABASE_URL in backend/.env)
and on the local SQLite file, with the same plain column types the other components use
(TEXT, REAL, INTEGER). One row per scored run: (team_id, run_id).
"""
import json
from datetime import datetime, timezone

from component2.db import Connection

SCHEMA = """
CREATE TABLE IF NOT EXISTS component3_alerts (
    team_id              TEXT NOT NULL,
    run_id               TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    run_ended_at         TEXT,
    anomaly_score        REAL,
    severity_score       REAL,
    severity_level       TEXT NOT NULL,
    is_issue             INTEGER NOT NULL,
    resolved             INTEGER NOT NULL DEFAULT 0,
    resolved_at          TEXT,
    resolved_by          TEXT,
    title                TEXT,
    evidence             TEXT,
    recommended_cause    TEXT,
    second_cause         TEXT,
    recommendation       TEXT,
    cause_probabilities  TEXT,
    n_epochs             INTEGER,
    drift_severity_score REAL,
    drift_source         TEXT,
    PRIMARY KEY (team_id, run_id)
);

CREATE INDEX IF NOT EXISTS ix_c3_alerts_team ON component3_alerts (team_id, created_at);
"""

COLUMNS = ["team_id", "run_id", "created_at", "run_ended_at", "anomaly_score", "severity_score",
           "severity_level", "is_issue", "resolved", "resolved_at", "resolved_by", "title", "evidence",
           "recommended_cause", "second_cause", "recommendation", "cause_probabilities", "n_epochs",
           "drift_severity_score", "drift_source"]


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(create_tables=False):
    conn = Connection()
    if create_tables:
        ensure_schema(conn)
    return conn


def ensure_schema(conn):
    for statement in SCHEMA.split(";"):
        if statement.strip():
            conn.execute(statement)
    conn.commit()


def save_alert(conn, team_id, result, run_ended_at=None):
    """Save one predict_run() result for a team. Scoring the same run again updates its scores
    but keeps whether someone already marked it resolved."""
    level = result["severity_level"]
    conn.execute(
        "INSERT INTO component3_alerts (team_id, run_id, created_at, run_ended_at, anomaly_score, "
        "severity_score, severity_level, is_issue, title, evidence, recommended_cause, second_cause, "
        "recommendation, cause_probabilities, n_epochs, drift_severity_score, drift_source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (team_id, run_id) DO UPDATE SET created_at = excluded.created_at, "
        "run_ended_at = excluded.run_ended_at, anomaly_score = excluded.anomaly_score, "
        "severity_score = excluded.severity_score, severity_level = excluded.severity_level, "
        "is_issue = excluded.is_issue, title = excluded.title, evidence = excluded.evidence, "
        "recommended_cause = excluded.recommended_cause, second_cause = excluded.second_cause, "
        "recommendation = excluded.recommendation, cause_probabilities = excluded.cause_probabilities, "
        "n_epochs = excluded.n_epochs, drift_severity_score = excluded.drift_severity_score, "
        "drift_source = excluded.drift_source",
        (team_id, result["run_id"], result.get("generated_at") or now_iso(), run_ended_at,
         result.get("anomaly_score"), result.get("severity_score"), level,
         1 if level in ("MEDIUM", "HIGH") else 0, result.get("title"), result.get("evidence"),
         result.get("recommended_cause"), result.get("second_cause"), result.get("recommendation"),
         json.dumps(result.get("root_cause_probabilities", {})), result.get("n_epochs"),
         result.get("drift_severity_score_used"), result.get("drift_source")),
    )
    conn.commit()


def list_alerts(conn, team_id):
    cur = conn.execute("SELECT " + ", ".join(COLUMNS) + " FROM component3_alerts WHERE team_id = ? "
                       "ORDER BY created_at DESC, run_id", (team_id,))
    return [dict(zip(COLUMNS, row)) for row in cur.fetchall()]


def set_resolved(conn, team_id, run_id, resolved, user_id=None):
    """Returns True if the alert exists."""
    cur = conn.execute(
        "UPDATE component3_alerts SET resolved = ?, resolved_at = ?, resolved_by = ? "
        "WHERE team_id = ? AND run_id = ?",
        (1 if resolved else 0, now_iso() if resolved else None, user_id if resolved else None, team_id, run_id))
    conn.commit()
    return cur.rowcount > 0
