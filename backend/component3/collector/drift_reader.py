"""Drift reader: get Component 2's drift score for a team and return ONE number for Component 3.

Component 2 writes one row per team, window and dimension (algorithm, metric, target) into the
`drift_results` table. Component 3's models take a single drift_severity_score between 0 and 1, so:

  1. for each dimension, take the latest window (at or before `as_of`, if given)
  2. ignore dimensions with no score (severity 'no_plan' or 'no_evidence')
  3. use the LARGEST remaining score: one badly drifting dimension is enough to matter

If no dimension has a score, the result is None and the caller decides what to do (assemble.build_run
falls back to 0.0 and records that in the output).

Reads only. Works on the shared PostgreSQL database (DATABASE_URL in backend/.env) and on the local
SQLite file the backend uses when DATABASE_URL is empty.

    python -m backend.component3.collector.drift_reader --list-teams
    python -m backend.component3.collector.drift_reader --team <team_id>
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BACKEND_DIR / ".env"
LOCAL_SQLITE = BACKEND_DIR / "component2" / "data" / "c2.db"

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
except ImportError:
    pass


def _is_postgres(url):
    return url.startswith(("postgres://", "postgresql://"))


def _query(sql_template, params, database_url=None, sqlite_path=None):
    """Run one read-only query; `sql_template` uses {p} where a parameter goes."""
    url = (database_url if database_url is not None else os.environ.get("DATABASE_URL", "")).strip()
    if _is_postgres(url):
        import psycopg
        with psycopg.connect(url, connect_timeout=10) as conn:
            conn.read_only = True
            return conn.execute(sql_template.format(p="%s"), params).fetchall()
    path = Path(sqlite_path) if sqlite_path else LOCAL_SQLITE
    if not path.exists():
        raise FileNotFoundError(f"No database: DATABASE_URL is empty and {path} does not exist.")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return conn.execute(sql_template.format(p="?"), params).fetchall()
    finally:
        conn.close()


def combine(rows):
    """rows: (dimension, window_end, drift_score, severity). Returns the summary dict."""
    latest = {}
    for dim, window_end, score, severity in rows:
        if dim not in latest or window_end > latest[dim][0]:
            latest[dim] = (window_end, score, severity)
    dims = {d: {"window_end": w, "drift_score": s, "severity": sev} for d, (w, s, sev) in sorted(latest.items())}
    scores = [v["drift_score"] for v in dims.values() if v["drift_score"] is not None]
    return {
        "drift_severity_score": round(max(scores), 4) if scores else None,
        "dimensions": dims,
        "source": "component2" if scores else "no_score",
    }


def read_drift(team_id, as_of=None, database_url=None, sqlite_path=None) -> dict:
    """Latest drift per dimension for a team, and the single combined score.
    `as_of` is an ISO text like '2026-10-18T17:00:00Z'; only windows ending at or before it count."""
    sql = "select dimension, window_end, drift_score, severity from drift_results where team_id = {p}"
    params = [team_id]
    if as_of:
        sql += " and window_end <= {p}"
        params.append(as_of)
    try:
        rows = _query(sql, params, database_url, sqlite_path)
    except Exception as e:                       # missing table, no connection...
        return {"drift_severity_score": None, "dimensions": {}, "source": f"unavailable: {type(e).__name__}"}
    out = combine(rows)
    out["team_id"] = team_id
    return out


def list_teams(database_url=None, sqlite_path=None):
    """(team_id, rows with a score) for each team in drift_results."""
    return _query("select team_id, count(drift_score) from drift_results group by team_id order by 2 desc, 1",
                  [], database_url, sqlite_path)


def main():
    p = argparse.ArgumentParser(description="Read Component 2's drift score for a team.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--team", help="team_id to read")
    g.add_argument("--list-teams", action="store_true", help="list team ids that have drift rows")
    p.add_argument("--as-of", help="only use windows ending at or before this ISO time")
    a = p.parse_args()
    if a.list_teams:
        for team_id, n in list_teams():
            print(f"{team_id}   rows with a score: {n}")
    else:
        print(json.dumps(read_drift(a.team, a.as_of), indent=2))


if __name__ == "__main__":
    main()
