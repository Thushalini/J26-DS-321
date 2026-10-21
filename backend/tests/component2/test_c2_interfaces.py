"""Component 2's interfaces on a throwaway SQLite database (never the shared one).

Checks the contract (api/contract.py, docs/interfaces.md) and the two live connections:
  C2 -> C4  summary + roles go through C4's c2_adapter into prediction features
  C2 -> C3  drift_results rows are read by C3's drift_reader as one drift number
and what each side does when C2's data is missing, invalid or late.
"""
import json
from datetime import timedelta

import pytest

from component2.api import contract

TEAM = "demo__interfaces"
A, B = "M-aaaaaa", "M-bbbbbb"


def ts(days_ago):
    return (contract.utc_now() - timedelta(days=days_ago)).strftime(contract.TS_FORMAT)


def seed(conn, days_ago=1, with_drift=True, with_roles=True):
    from component2.db import save_activities, save_drift, save_labels, save_roles

    work = [("commit:1", A, "commit", "Clean missing values", "data_cleaning"),
            ("commit:2", A, "commit", "Train random forest", "model_development"),
            ("nb:1", B, "notebook_cell", "df.describe()", "eda"),
            ("chat:1", B, "chat", "I'll do the plots", "eda")]
    save_activities(conn, [{"activity_id": f"{TEAM}:{aid}", "team_id": TEAM, "source": src, "member_id": m,
                            "ts": ts(days_ago + n), "text": text, "meta": {}}
                           for n, (aid, m, src, text, _) in enumerate(work)])
    save_labels(conn, "model", [(f"{TEAM}:{aid}", stage, 0.9) for aid, _, _, _, stage in work])
    if with_drift:
        save_drift(conn, [{"team_id": TEAM, "window_end": ts(days_ago), "dimension": d, "planned": p,
                           "seen_top": p, "drift_score": s, "severity": sev, "explanation": "test"}
                          for d, p, s, sev in (("target", "churn", 0.1, "none"),
                                               ("algorithm", "random forest", 0.45, "moderate"),
                                               ("metric", None, None, "no_plan"))])
    if with_roles:
        save_roles(conn, [{"team_id": TEAM, "member_id": m, "week": "2026-W42", "stage_mix": {"eda": 1.0},
                           "chat": {}, "dominant_stage": "eda"} for m in (A, B)])


# ---------- Output contract ----------

def test_summary_follows_the_contract(conn):
    from component2.api import service
    seed(conn)
    summary = service.team_summary(conn, TEAM)
    assert contract.check_summary(summary) == []
    assert summary["version"] == contract.CONTRACT_VERSION
    assert summary["data_status"]["status"] == "ok"


def test_contract_catches_a_broken_payload():
    assert "missing 'data_status'" in contract.check_summary({"component": "C2"})
    assert contract.check_summary(None) == ["summary is not an object"]


# ---------- Connection C2 -> C4 ----------

def test_c2_summary_and_roles_become_c4_features(conn):
    from component2.api import service
    from component4.m1_feature_integration import c2_adapter
    seed(conn)
    part = c2_adapter.project_features(service.team_summary(conn, TEAM), service.member_roles(conn, TEAM, "30d"))
    assert part["features"]["drift_score"] == pytest.approx(0.45)         # highest scored dimension
    assert part["context"]["drift_driver"] == "algorithm"
    assert 0 <= part["features"]["contribution_balance"] <= 1


def test_c4_gets_none_when_c2_has_no_drift_yet(conn):
    from component2.api import service
    from component4.m1_feature_integration import c2_adapter
    seed(conn, with_drift=False)
    summary = service.team_summary(conn, TEAM)
    assert summary["data_status"]["status"] == "incomplete" and "drift" in summary["data_status"]["missing"]
    assert c2_adapter.project_features(summary)["features"]["drift_score"] is None   # C4 fills a typical value


# ---------- Connection C2 -> C3 ----------

def test_c3_reads_the_same_drift_number(conn, tmp_path):
    from component3.collector.drift_reader import read_drift
    seed(conn)
    out = read_drift(TEAM, database_url="", sqlite_path=tmp_path / "c2_test.db")
    assert out["drift_severity_score"] == pytest.approx(0.45) and out["source"] == "component2"


def test_c3_reports_no_score_when_c2_has_no_drift(conn, tmp_path):
    from component3.collector.drift_reader import read_drift
    seed(conn, with_drift=False)
    out = read_drift(TEAM, database_url="", sqlite_path=tmp_path / "c2_test.db")
    assert out["drift_severity_score"] is None and out["source"] == "no_score"


def test_c3_survives_c2_database_being_unavailable(tmp_path):
    from component3.collector.drift_reader import read_drift
    out = read_drift(TEAM, database_url="", sqlite_path=tmp_path / "missing.db")
    assert out["drift_severity_score"] is None and out["source"].startswith("unavailable")


# ---------- Over HTTP ----------

def test_summary_endpoint(conn):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from component2.api.router import router
    seed(conn)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    r = client.get(f"/c2/teams/{TEAM}/summary")
    assert r.status_code == 200 and contract.check_summary(r.json()) == []
    assert client.get("/c2/teams/nobody/summary").status_code == 404
    assert client.get(f"/c2/teams/{TEAM}/summary?days=0").status_code == 422


# ---------- Delayed data ----------

def test_old_activity_is_marked_stale(conn):
    from component2.api import service
    seed(conn, days_ago=contract.STALE_AFTER_DAYS + 5)
    status = service.team_summary(conn, TEAM)["data_status"]
    assert status["status"] == "stale" and status["days_since_last_activity"] > contract.STALE_AFTER_DAYS


# ---------- Invalid input ----------

@pytest.mark.parametrize("change, reason", [
    ({"ts": "21/10/2026"}, "not in"),
    ({"ts": "2099-01-01T00:00:00Z"}, "future"),
    ({"text": "   "}, "empty text"),
    ({"source": "email"}, "unknown source"),
    ({"team_id": ""}, "missing team_id"),
])
def test_bad_activities_are_rejected(change, reason):
    row = {"activity_id": "chat:t:m1", "team_id": "t", "source": "chat", "member_id": A,
           "ts": ts(1), "text": "hello"}
    assert contract.check_activity(row) is None
    assert reason in contract.check_activity({**row, **change})


def test_chat_file_skips_bad_rows(tmp_path):
    from component2.chat.load_chat import load_file
    path = tmp_path / "team_x.csv"
    path.write_text("msg_id,timestamp,sender,message,reply_to\n"
                    f"m1,{ts(1)},{A},Cleaned the data,\n"
                    f"m2,not-a-date,{A},Bad time,\n"
                    f"m3,{ts(1)},{B},,\n"
                    f",{ts(1)},{B},No id,\n", encoding="utf-8")
    rows, rejected = load_file(path)
    assert [r["activity_id"] for r in rows] == ["chat:team_x:m1"]
    assert len(rejected) == 3


def test_blueprint_is_cleaned():
    assert contract.clean_plan({"target": " churn ", "algorithm": {"name": "XGBoost"}, "metric": 5}) == \
        {"target": "churn", "algorithm": "XGBoost", "metric": None}
    assert contract.clean_plan(None) == {"target": None, "algorithm": None, "metric": None}


def test_component1_blueprint_is_accepted_only_for_settled_fields():
    c1 = {"fields": {"target_variable": {"value": "churn", "status": "confirmed"},
                     "algorithm": {"value": "XGBoost", "status": "conflicting"},
                     "evaluation_metric": {"value": "F1", "status": "known"}}}
    assert contract.clean_plan(c1) == {"target": "churn", "algorithm": None, "metric": "F1"}


def test_corrupt_blueprint_file_is_kept_and_reported(conn, tmp_path, monkeypatch):
    from component2.drift import detect_drift
    seed(conn, with_drift=False)
    monkeypatch.setattr(detect_drift, "BLUEPRINT_DIR", tmp_path)
    bad = tmp_path / f"{TEAM}.json"
    bad.write_text("{not json", encoding="utf-8")
    results = detect_drift.detect_team(conn, TEAM, 30)
    assert {r["severity"] for r in results} == {"no_plan"} and results[0]["blueprint"] == "invalid_file"
    assert bad.read_text(encoding="utf-8") == "{not json"          # not overwritten by an inferred plan
