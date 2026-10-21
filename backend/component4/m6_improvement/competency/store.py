"""Saved score sheets per group member: only module code, credits and grade, never shown to anyone."""
import json
from datetime import datetime, timezone

TABLE = "component4_competency_sheets"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save(conn, team_id: str, user_id: str, modules: list[dict], uploaded_by: str) -> None:
    conn.execute(f"INSERT INTO {TABLE} (team_id, user_id, modules, uploaded_by, uploaded_at) VALUES (?, ?, ?, ?, ?) "
                 "ON CONFLICT (team_id, user_id) DO UPDATE SET modules = excluded.modules, "
                 "uploaded_by = excluded.uploaded_by, uploaded_at = excluded.uploaded_at",
                 (team_id, user_id, json.dumps(modules), uploaded_by, _now()))
    conn.commit()


def delete(conn, team_id: str, user_id: str) -> None:
    conn.execute(f"DELETE FROM {TABLE} WHERE team_id = ? AND user_id = ?", (team_id, user_id))
    conn.commit()


def saved_dates(conn, team_id: str) -> dict[str, str]:
    return dict(conn.execute(f"SELECT user_id, uploaded_at FROM {TABLE} WHERE team_id = ?", (team_id,)).fetchall())


def load_all(conn, team_id: str) -> dict[str, list[dict]]:
    return {user_id: json.loads(modules) for user_id, modules in
            conn.execute(f"SELECT user_id, modules FROM {TABLE} WHERE team_id = ?", (team_id,)).fetchall()}
