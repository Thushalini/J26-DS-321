"""Drift for multi-model projects: one plan per folder ("part"), ignored vocabulary files,
and one-model projects working exactly as before. MiniLM is replaced by fake_embed."""
import json

import pytest

from component2.api import contract, service
from component2.db import save_drift
from component2.drift import detect_drift
from component2.drift.detect_drift import is_ignored, part_of, split_by_part

TEAM = "demo__multi"
A, B = "M-aaaaaa", "M-bbbbbb"
PARTS = [{"name": "C2 classifier", "path": "backend/component2",
          "target": "stage", "algorithm": "logistic regression", "metric": "f1"},
         {"name": "C4 prediction", "path": "backend/component4",
          "target": "success", "algorithm": "random forest", "metric": "roc auc"}]
IGNORE = ["backend/component2/drift/mentions.py", "*/tests/*"]


@pytest.fixture(autouse=True)
def no_model(monkeypatch, fake_embed):
    monkeypatch.setattr(detect_drift, "embed", fake_embed)


def code(aid, member, ts, path, text):
    return (aid, member, "code_change", ts, f"File: {path}\n{text}", None)


def work(add):
    rows = [code("k1", A, "2026-10-10T10:00:00Z", "backend/component2/classify/train_model.py",
                 "clf = LogisticRegression(max_iter=5000)\nscoring='f1_macro'"),
            code("k2", A, "2026-10-11T10:00:00Z", "backend/component2/drift/mentions.py",
                 "'xgboost': [...], 'bert': [...], 'yolo': [...]"),                    # ignored: only a list
            code("k3", B, "2026-10-12T10:00:00Z", "backend/component4/m2_prediction/models.py",
                 "RandomForestClassifier(n_estimators=300)"),
            code("k4", B, "2026-10-13T10:00:00Z", "backend/component4/m2_prediction/models.py",
                 "XGBClassifier(); StackingClassifier(...)"),
            code("k5", B, "2026-10-14T10:00:00Z", "backend/component4/tests/test_prediction.py",
                 "LGBMClassifier()"),                                                    # ignored: tests
            ("c1", A, "commit", "2026-10-15T10:00:00Z", "Tune LogisticRegression C", None),
            ("c2", B, "commit", "2026-10-16T10:00:00Z", "Try lightgbm everywhere", None)]
    add(TEAM, rows, meta={"k1": {"file": "backend/component2/classify/train_model.py"},
                          "k2": {"file": "backend/component2/drift/mentions.py"},
                          "k3": {"file": "backend/component4/m2_prediction/models.py"},
                          "k4": {"file": "backend/component4/m2_prediction/models.py"},
                          "k5": {"file": "backend/component4/tests/test_prediction.py"},
                          "c1": {"files": ["backend/component2/classify/train_model.py"]},
                          "c2": {"files": ["backend/component2/config.py", "backend/component4/common/config.py"]}})


# ---------- The plan ----------

def test_clean_blueprint_keeps_valid_parts_only():
    bp = contract.clean_blueprint({
        "algorithm": "xgboost",
        "parts": [{"name": " C2 ", "path": "./backend\\component2/", "algorithm": "Logistic Regression"},
                  {"name": "", "path": "x"},                         # no name
                  {"name": "C2", "path": "other"},                   # same name twice
                  {"name": "bad::name", "path": "y"},                # separator used in drift rows
                  "not a part"],
        "ignore": ["./backend/tests/", "", 5]})
    assert bp["plan"]["algorithm"] == "xgboost"
    assert bp["parts"] == [{"name": "C2", "path": "backend/component2",
                            "plan": {"target": None, "algorithm": "Logistic Regression", "metric": None}}]
    assert bp["ignore"] == ["backend/tests"]


def test_one_model_blueprint_has_no_parts():
    assert contract.clean_blueprint({"target": "churn"}) == \
        {"plan": {"target": "churn", "algorithm": None, "metric": None}, "parts": [], "ignore": []}


def test_files_go_to_the_deepest_matching_folder():
    parts = [{"name": "backend", "path": "backend"}, {"name": "C2", "path": "backend/component2"}]
    assert part_of("backend/component2/x.py", parts) == "C2"
    assert part_of("backend/main.py", parts) == "backend"
    assert part_of("backend_old/x.py", parts) is None              # a folder name, not a text prefix
    assert part_of("frontend/app.tsx", parts) is None


def test_ignore_patterns():
    assert is_ignored("backend/tests/test_x.py", ["*/tests/*"])
    assert is_ignored("backend/component2/drift/mentions.py", ["backend/component2/drift"])   # a folder
    assert not is_ignored("backend/component2/drift_extra.py", ["backend/component2/drift"])


def test_commits_go_to_the_part_most_of_their_files_are_in():
    parts = contract.clean_blueprint({"parts": PARTS})["parts"]
    def commit(*files):
        return ("2026-10-10T10:00:00Z", "msg", "commit", json.dumps({"files": list(files)}))
    groups = split_by_part([commit("backend/component2/a.py", "backend/component2/b.py", "backend/component4/c.py"),
                            commit("backend/component2/a.py", "backend/component4/c.py"),        # tie
                            commit("backend/tests/test_a.py"),                                     # only ignored
                            ("2026-10-10T10:00:00Z", "hi", "chat", "{}")], parts, ["*/tests/*"])
    assert len(groups["C2 classifier"]) == 1
    assert len(groups[None]) == 2                                  # the tie and the chat message
    assert sum(len(g) for g in groups.values()) == 3


# ---------- Detection ----------

def test_each_part_is_compared_with_its_own_plan(conn, add):
    work(add)
    plan = {"target": None, "algorithm": None, "metric": None, "parts": PARTS, "ignore": IGNORE}
    results = {r["dimension"]: r for r in detect_drift.detect_team(conn, TEAM, 30, planned=plan)}
    c2 = results["C2 classifier::algorithm"]
    assert c2["drift_score"] == 0.0 and c2["part"] == "C2 classifier"      # mentions.py's list is ignored
    c4 = results["C4 prediction::algorithm"]
    assert c4["drift_score"] == pytest.approx(0.667, abs=0.001)          # rf on plan; xgboost + stacking off
    assert "lightgbm" not in (c4["explanation"] or "")                  # the tests folder is ignored
    assert results["C2 classifier::metric"]["drift_score"] == 0.0          # scoring='f1_macro' is the planned f1
    # the commit spread over both parts is compared with the project plan, which is empty
    assert results["algorithm"]["severity"] == "no_plan"


def test_without_parts_the_whole_repo_is_one_plan_as_before(conn, add):
    work(add)
    results = {r["dimension"]: r for r in detect_drift.detect_team(
        conn, TEAM, 30, planned={"target": None, "algorithm": "logistic regression", "metric": None})}
    assert set(results) == {"target", "algorithm", "metric"}
    # nothing ignored: xgboost, bert, yolo (the keyword list), rf, stacking, lightgbm all count as drift
    assert results["algorithm"]["drift_score"] > 0.5


def test_blueprint_file_with_parts(conn, add, tmp_path, monkeypatch):
    monkeypatch.setattr(detect_drift, "BLUEPRINT_DIR", tmp_path)
    (tmp_path / f"{TEAM}.json").write_text(json.dumps({"parts": PARTS, "ignore": IGNORE}), encoding="utf-8")
    work(add)
    results = {r["dimension"]: r for r in detect_drift.detect_team(conn, TEAM, 30)}
    assert results["C2 classifier::algorithm"]["blueprint"] == "file"
    assert results["C4 prediction::metric"]["severity"] == "no_evidence"


# ---------- Output for C3 / C4 ----------

def test_summary_shows_parts_and_keeps_one_result_per_decision(conn, add):
    work(add)
    plan = {"algorithm": "logistic regression", "parts": PARTS, "ignore": IGNORE}
    save_drift(conn, detect_drift.detect_team(conn, TEAM, 30, planned=plan))
    out = service.drift(conn, TEAM)
    assert [d["dimension"] for d in out["dimensions"]] == ["target", "algorithm", "metric"]
    algorithm = out["dimensions"][1]
    # the worst of project and parts: the commit spread over both parts mentions lightgbm, off the project plan
    assert algorithm["part"] is None and algorithm["drift_score"] == 1.0
    metric = out["dimensions"][2]
    assert metric["part"] == "C2 classifier" and metric["drift_score"] == 0.0   # the only scored metric
    assert {p["part"] for p in out["parts"]} == {"C2 classifier", "C4 prediction"}
    summary = service.team_summary(conn, TEAM)
    assert contract.check_summary(summary) == []


def test_c4_and_c3_read_the_worst_part(conn, add, tmp_path):
    from component3.collector.drift_reader import read_drift
    from component4.m1_feature_integration import c2_adapter
    work(add)
    save_drift(conn, detect_drift.detect_team(conn, TEAM, 30, planned={"parts": PARTS, "ignore": IGNORE}))
    features = c2_adapter.project_features(service.team_summary(conn, TEAM))
    assert features["features"]["drift_score"] == pytest.approx(0.667, abs=0.001)
    assert read_drift(TEAM, database_url="", sqlite_path=tmp_path / "c2_test.db")["drift_severity_score"] == \
        pytest.approx(0.667, abs=0.001)


def test_app_blueprint_with_parts_is_used_by_the_sync(conn, add, monkeypatch, fake_embed):
    from buddy import store, sync
    from component2.roles import role_tracker
    monkeypatch.setattr(role_tracker, "embed", fake_embed)
    store.ensure_tables(conn)
    work(add)
    conn.execute("INSERT INTO blueprints (team_id, target, algorithm, metric, notes, updated_by, updated_at, "
                 "parts, ignore_paths) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 (TEAM, None, None, None, None, None, "2026-10-21T00:00:00Z", json.dumps(PARTS), json.dumps(IGNORE)))
    conn.commit()
    sync.analyse_team(conn, TEAM)
    assert {p["part"] for p in service.drift(conn, TEAM)["parts"]} == {"C2 classifier", "C4 prediction"}


def test_blueprint_api_saves_parts_and_keeps_them_when_not_sent(conn, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from buddy import store
    from buddy.router import router
    monkeypatch.setattr(store, "_ready", False)               # create the app tables in the test database
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    r = client.post("/app/auth/register", json={"email": "lead@t.io", "name": "Lead", "password": "password123"})
    lead = {"Authorization": "Bearer " + r.json()["token"]}
    team_id = client.post("/app/teams", json={"name": "Multi"}, headers=lead).json()["team_id"]
    url = f"/app/teams/{team_id}/blueprint"

    saved = client.put(url, json={"algorithm": "xgboost", "parts": PARTS[:1], "ignore": ["*/tests/*"]}, headers=lead).json()
    assert saved["parts"] == [{"name": "C2 classifier", "path": "backend/component2", "target": "stage",
                               "algorithm": "logistic regression", "metric": "f1"}]
    assert saved["ignore"] == ["*/tests/*"]
    again = client.put(url, json={"algorithm": "lightgbm"}, headers=lead).json()      # an older client
    assert again["algorithm"] == "lightgbm" and again["parts"] == saved["parts"]
    cleared = client.put(url, json={"parts": [], "ignore": []}, headers=lead).json()
    assert cleared["parts"] == [] and cleared["ignore"] == []
    too_many = client.put(url, json={"parts": [{"name": f"p{n}", "path": f"p{n}"} for n in range(11)]}, headers=lead)
    assert too_many.status_code == 422
