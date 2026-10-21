"""Real results the group records after each milestone (self-reported), kept to check Buddy's predictions
against what actually happened."""
import uuid
from datetime import datetime, timezone

TABLE = "component4_milestone_results"


def add(conn, team_id: str, milestone: str, milestone_date: str, result: str, recorded_by: str) -> None:
    conn.execute(f"INSERT INTO {TABLE} (team_id, result_id, milestone, milestone_date, result, recorded_by, recorded_at) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (team_id, uuid.uuid4().hex, milestone, milestone_date, result, recorded_by,
                  datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.commit()


def remove(conn, team_id: str, result_id: str) -> None:
    conn.execute(f"DELETE FROM {TABLE} WHERE team_id = ? AND result_id = ?", (team_id, result_id))
    conn.commit()


def listed(conn, team_id: str) -> list[dict]:
    rows = conn.execute(f"SELECT result_id, milestone, milestone_date, result, recorded_at FROM {TABLE} "
                        "WHERE team_id = ? ORDER BY milestone_date DESC", (team_id,)).fetchall()
    return [{"result_id": rid, "milestone": name, "date": day, "result": result, "recorded_at": at}
            for rid, name, day, result, at in rows]


def paired_with_predictions(conn) -> list[dict]:
    """Every recorded result with the group's last saved prediction on or before the milestone date."""
    rows = conn.execute(f"SELECT r.team_id, r.milestone, r.milestone_date, r.result, "
                        "(SELECT h.probability FROM component4_prediction_history h WHERE h.team_id = r.team_id "
                        "AND h.day <= r.milestone_date ORDER BY h.day DESC LIMIT 1) "
                        f"FROM {TABLE} r ORDER BY r.milestone_date").fetchall()
    return [{"team_id": team, "milestone": name, "date": day, "result": result, "probability_before": probability}
            for team, name, day, result, probability in rows]
