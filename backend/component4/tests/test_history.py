import sqlite3
from pathlib import Path

from component4.m6_improvement import history

MIGRATION = Path(__file__).resolve().parents[2] / "buddy" / "migrations" / "0005_component4_prediction_history.sql"


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.executescript(MIGRATION.read_text(encoding="utf-8"))
    return conn


def _result(probability):
    return {"evaluated_at": "2026-10-21T10:00:00+00:00", "probability": probability, "model_version": "v1"}


def test_one_saved_prediction_per_day():
    conn = _conn()
    history.record(conn, "t", _result(40.0), {"open_high_issues": 3.0}, day="2026-10-20")
    history.record(conn, "t", _result(45.0), {"open_high_issues": 3.0}, day="2026-10-20")
    assert [row["probability"] for row in history.recent(conn, "t", "v1")] == [45.0]


def test_changes_since_the_previous_day_list_only_measured_signals_that_moved():
    conn = _conn()
    history.record(conn, "t", _result(40.0), {"open_high_issues": 3.0, "drift_score": 0.2, "data_quality": None},
                   day="2026-10-20")
    history.record(conn, "t", _result(52.5), {"open_high_issues": 1.0, "drift_score": 0.21, "data_quality": 0.9},
                   day="2026-10-21")
    change = history.changes(history.recent(conn, "t", "v1"))
    assert change["probability_change"] == 12.5
    assert [s["id"] for s in change["signals"]] == ["open_high_issues"]


def test_other_model_versions_are_left_out():
    conn = _conn()
    history.record(conn, "t", {**_result(40.0), "model_version": "old"}, {}, day="2026-10-20")
    history.record(conn, "t", _result(50.0), {}, day="2026-10-21")
    assert len(history.recent(conn, "t", "v1")) == 1 and history.changes(history.recent(conn, "t", "v1")) is None
