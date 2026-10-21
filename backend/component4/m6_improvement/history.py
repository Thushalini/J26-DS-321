"""Re-evaluation over time: one saved prediction per team per day, and what changed since the previous one.

Only measured signals are saved, so "before -> after" never compares typical filled-in values.
"""
import json
from datetime import datetime, timezone

from ..common import config
from ..synthetic.feature_spec import get

TABLE = "component4_prediction_history"


def record(conn, team_id: str, result: dict, features: dict, day: str | None = None) -> None:
    day = day or datetime.now(timezone.utc).date().isoformat()
    measured = {name: value for name, value in features.items() if value is not None}
    conn.execute(f"INSERT INTO {TABLE} (team_id, day, evaluated_at, probability, model_version, features) "
                 "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (team_id, day) DO UPDATE SET "
                 "evaluated_at = excluded.evaluated_at, probability = excluded.probability, "
                 "model_version = excluded.model_version, features = excluded.features",
                 (team_id, day, result["evaluated_at"], result["probability"], result["model_version"],
                  json.dumps(measured)))
    conn.commit()


def recent(conn, team_id: str, model_version: str) -> list[dict]:
    rows = conn.execute(f"SELECT day, evaluated_at, probability, features FROM {TABLE} "
                        "WHERE team_id = ? AND model_version = ? ORDER BY day DESC LIMIT ?",
                        (team_id, model_version, config.HISTORY["points"])).fetchall()
    return [{"day": day, "evaluated_at": at, "probability": probability, "features": json.loads(features)}
            for day, at, probability, features in reversed(rows)]


def changes(rows: list[dict]) -> dict | None:
    if len(rows) < 2:
        return None
    before, after = rows[-2], rows[-1]
    limits = config.HISTORY["changed_at_least"]
    signals = [
        {"id": name, "label": config.FEATURE_LABELS.get(name, name), "before": round(before["features"][name], 3),
         "after": round(value, 3)}
        for name, value in after["features"].items()
        if name in before["features"] and abs(value - before["features"][name]) >= limits[get(name).kind]
    ]
    return {"since": before["evaluated_at"], "probability_change": round(after["probability"] - before["probability"], 1),
            "signals": signals}
