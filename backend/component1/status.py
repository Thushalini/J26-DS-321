"""See what Component 1 has stored, without opening the database by hand. Read-only.

Run from the backend folder:
    python -m component1.status                 lists the teams Component 1 knows about
    python -m component1.status TEAM_ID         shows one team's blueprint, Buddy's messages and the old blueprints row

It says which database it is looking at, and never prints the database URL or password.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .extraction.gemini_provider import load_env_file
from .plan import build_plan

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _rows(conn: Any, sql: str, params: tuple = ()) -> list[tuple]:
    return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def describe(conn: Any, team_id: str | None = None) -> list[str]:
    lines: list[str] = []
    try:
        teams = _rows(conn, "SELECT team_id, COUNT(*), MAX(version) FROM c1_events GROUP BY team_id ORDER BY team_id")
    except Exception:
        return ["Component 1 has stored nothing yet. Its tables are created when the first chat message is handled."]

    if team_id is None:
        if not teams:
            return ["Component 1 has stored nothing yet. Send a chat message while the backend is running."]
        lines.append("Teams Component 1 has heard from:")
        for tid, n_events, version in teams:
            said = _rows(conn, "SELECT COUNT(*) FROM c1_agent_messages WHERE team_id = ?", (tid,))[0][0]
            seen = _rows(conn, "SELECT COUNT(*) FROM c1_processed WHERE team_id = ?", (tid,))[0][0]
            lines.append(f"  {tid}: {n_events} saved changes (version {version}), Buddy said {said} things, handled {seen} chat messages")
        lines.append("Run again with a TEAM_ID to see its blueprint.")
        return lines

    plan = build_plan(conn, team_id, include_evidence=False)
    r = plan["readiness"]
    lines.append(f"Team {team_id}: blueprint version {plan['version']}, {r['confirmed']} of {r['total']} fields confirmed")
    lines.append("What Component 1 knows (this is what the Blueprint tab shows):")
    for f in plan["fields"]:
        value = f["value"] if f["value"] is not None else "-"
        band = f["confidence_band"] or "-"
        extra = f"   candidates: {', '.join(f['candidates'])}" if f.get("candidates") else ""
        lines.append(f"  {f['name']:<18} {value:<26} {f['status']:<12} {band}{extra}")
    lines.append(f"Ready to build? {r['complete']}.  Next questions Buddy may ask about: {', '.join(plan['next_gaps']) or '-'}")

    messages = _rows(conn, "SELECT ts, text FROM c1_agent_messages WHERE team_id = ? ORDER BY ts DESC", (team_id,))
    lines.append(f"Buddy has said {len(messages)} things. The latest:")
    for ts, text in messages[:3]:
        lines.append(f"  {ts}  {text.splitlines()[0][:100]}")

    try:
        row = _rows(conn, "SELECT target, algorithm, metric, updated_by FROM blueprints WHERE team_id = ?", (team_id,))
        if row:
            target, algorithm, metric, by = row[0]
            lines.append(f"The OLD three-value blueprint (read by Component 2): target={target!r}, algorithm={algorithm!r}, metric={metric!r}, updated by {by!r}")
        else:
            lines.append("The OLD three-value blueprint has no row for this team. That is normal until a field becomes KNOWN or CONFIRMED.")
    except Exception:
        lines.append("Could not read the old blueprints table.")
    return lines


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    load_env_file(BACKEND_DIR / ".env")
    kind = "Postgres (Supabase), the SHARED database" if os.environ.get("DATABASE_URL") else "SQLite fallback (local only)"
    print(f"Database: {kind}")
    try:
        from component2.db import connect
        conn = connect()
    except Exception as exc:
        print(f"Could not connect: {type(exc).__name__}")
        return 1
    try:
        for line in describe(conn, argv[0] if argv else None):
            print(line)
    finally:
        close = getattr(conn, "close", None)
        if close:
            close()
    return 0


if __name__ == "__main__":
    sys.exit(main())