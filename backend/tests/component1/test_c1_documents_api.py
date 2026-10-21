"""Tests for component1/api/documents.py and the plan's suggestions, through FastAPI on SQLite.

Sign-in is replaced by a fixed test user; the app tables are stand-ins with the real column
names that the access checks read. All names and text are invented.
"""
import json
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

import component1.hook as hook
from buddy.auth import current_user, get_conn
from component1.api.router import router
from component1.extraction.hybrid import make_hybrid_llm
from component1.service import AgentService
from component1.state.features import FEATURE_ORDER
from component1.state.simulate import FEATURE_NAMES, generate_dataset


def small_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


BUNDLE = small_bundle()
PLAN = b"We will use the Kaggle tourism dataset.\n\nWe will predict tourist arrivals using ARIMA and report RMSE."
USERS = {
    "lead": {"user_id": "u1", "name": "Ayesha", "role": "student"},
    "member": {"user_id": "u2", "name": "Bimal", "role": "student"},
    "supervisor": {"user_id": "u3", "name": "Dr. Silva", "role": "supervisor"},
    "examiner": {"user_id": "u4", "name": "Examiner", "role": "examiner"},
    "outsider": {"user_id": "u5", "name": "Other", "role": "student"},
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / "app.db"

    def connect():
        return sqlite3.connect(str(path), check_same_thread=False)

    c = connect()
    c.executescript(
        """
        CREATE TABLE teams (team_id TEXT PRIMARY KEY, name TEXT, is_research INTEGER);
        CREATE TABLE team_members (team_id TEXT, user_id TEXT, team_role TEXT, member_code TEXT);
        CREATE TABLE panel_reviews (user_id TEXT, team_id TEXT, started_at TEXT, expires_at TEXT);
        CREATE TABLE blueprints (team_id TEXT PRIMARY KEY, target TEXT, algorithm TEXT, metric TEXT, notes TEXT,
                                 updated_by TEXT, updated_at TEXT);
        CREATE TABLE activities (activity_id TEXT, team_id TEXT);
        INSERT INTO teams VALUES ('t1', 'Team One', 0), ('r1', 'Research', 1);
        INSERT INTO team_members VALUES ('t1', 'u1', 'lead', 'M-1'), ('t1', 'u2', 'member', 'M-2'),
                                        ('t1', 'u3', 'supervisor', NULL);
        """
    )
    c.commit()
    c.close()

    svc = AgentService(bundle=BUNDLE, llm=make_hybrid_llm(None, mode="rules"), env_file=None, conn_factory=connect)
    monkeypatch.setattr(hook, "_service", svc)
    # "tables created" is remembered per process (one database in the app); each test has a new one
    import component1.api.documents as documents_api
    import component1.api.router as router_api
    monkeypatch.setattr(documents_api, "_tables_ready", False)
    monkeypatch.setattr(router_api, "_tables_ready", False)

    app = FastAPI()
    app.include_router(router)
    who = {"user": USERS["member"]}

    def fake_conn():
        conn = connect()
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_conn] = fake_conn
    app.dependency_overrides[current_user] = lambda: who["user"]
    test_client = TestClient(app)
    test_client.who = who
    return test_client


def as_(client, role):
    client.who["user"] = USERS[role]
    return client


def upload(client, data=PLAN, name="Plan.txt", team="t1"):
    return client.post(f"/app/teams/{team}/documents", files={"file": (name, data, "text/plain")})


def field(plan, name):
    return next(f for f in plan["fields"] if f["name"] == name)


# ------------------------------------------------------------------ upload
def test_a_member_uploads_and_the_plan_shows_suggestions_with_their_source(client):
    r = upload(client)
    assert r.status_code == 200 and r.json()["document"]["status"] == "processing"

    docs = client.get("/app/teams/t1/documents").json()
    assert docs["can_upload"] and docs["documents"][0]["status"] == "done"

    plan = client.get("/app/teams/t1/plan").json()
    (s,) = field(plan, "algorithm")["suggestions"]
    assert (s["value"], s["source"], s["text"]) == ("ARIMA", "Plan.txt", "ARIMA")
    assert field(plan, "algorithm")["status"] == "missing"        # a suggestion is not a value yet
    assert plan["can_edit"] and plan["documents"][0]["filename"] == "Plan.txt"


@pytest.mark.parametrize("name, data, code", [("slides.pptx", b"x", 415), ("Plan.txt", b"", 400)])
def test_unusable_uploads_are_refused_before_any_work(client, name, data, code):
    assert upload(client, data=data, name=name).status_code == code
    assert client.get("/app/teams/t1/documents").json()["documents"] == []


def test_research_teams_and_outsiders_cannot_upload(client):
    assert upload(client, team="r1").status_code in (400, 403)
    assert upload(as_(client, "outsider")).status_code == 403


# ------------------------------------------------------------ who sees what
def test_a_supervisor_sees_suggestions_without_the_text_and_cannot_decide(client):
    upload(client)
    sup = as_(client, "supervisor")
    assert upload(sup).status_code == 403
    plan = sup.get("/app/teams/t1/plan").json()
    (s,) = field(plan, "algorithm")["suggestions"]
    assert "text" not in s and not plan["can_edit"]
    assert sup.post(f"/app/teams/t1/plan/candidates/{s['fact_id']}/accept").status_code == 403


def test_an_examiner_does_not_see_file_names(client):
    upload(client)
    ex = as_(client, "examiner")
    assert ex.get("/app/teams/t1/documents").json()["documents"][0]["filename"] == "Document 1"
    plan = ex.get("/app/teams/t1/plan").json()
    assert plan["documents"] is None and "suggestions" not in field(plan, "algorithm")


# ---------------------------------------------------------- accept/dismiss
def test_accepting_moves_the_value_into_the_plan(client):
    upload(client)
    fact_id = field(client.get("/app/teams/t1/plan").json(), "algorithm")["suggestions"][0]["fact_id"]
    r = client.post(f"/app/teams/t1/plan/candidates/{fact_id}/accept")
    assert r.status_code == 200 and r.json()["ok"]
    algo = field(r.json()["plan"], "algorithm")
    assert algo["value"] == "ARIMA" and algo["status"] != "missing" and algo["suggestions"] == []
    assert [e["speaker"] for e in algo["evidence"]] == ["doc:Plan.txt", "M-2"]
    assert client.post(f"/app/teams/t1/plan/candidates/{fact_id}/accept").status_code == 409


def test_dismissing_removes_the_suggestion(client):
    upload(client)
    fact_id = field(client.get("/app/teams/t1/plan").json(), "dataset")["suggestions"][0]["fact_id"]
    r = client.post(f"/app/teams/t1/plan/candidates/{fact_id}/dismiss")
    assert r.status_code == 200
    assert field(r.json()["plan"], "dataset")["suggestions"] == []
    assert field(r.json()["plan"], "dataset")["status"] == "missing"


def test_removing_a_document_removes_its_open_suggestions(client):
    upload(client)
    doc_id = client.get("/app/teams/t1/documents").json()["documents"][0]["doc_id"]
    assert client.delete(f"/app/teams/t1/documents/{doc_id}").status_code == 200
    assert field(client.get("/app/teams/t1/plan").json(), "algorithm")["suggestions"] == []
    assert client.delete(f"/app/teams/t1/documents/{doc_id}").status_code == 404


# ---------------------------------------------------------------- history
def test_the_history_route_shows_who_changed_what_with_quotes_only_for_the_team(client):
    hook._service.handle_chat_message("t1", "m1", "M-1", "Let's use XGBoost for this.", "2026-10-21T10:00:00Z")
    upload(client)
    fact_id = field(client.get("/app/teams/t1/plan").json(), "algorithm")["suggestions"][0]["fact_id"]
    client.post(f"/app/teams/t1/plan/candidates/{fact_id}/accept")

    r = client.get("/app/teams/t1/plan/history?field=algorithm").json()
    assert r["total"] == 2
    newest, first = r["entries"]
    assert first["kind"] == "proposed" and first["value"] == "XGBoost" and first["text"]
    assert newest["kind"] == "accepted_document" and newest["value"] == "ARIMA"
    assert newest["status_after"] == "conflicting"                      # chat said XGBoost, the document ARIMA

    plan = client.get("/app/teams/t1/plan").json()
    assert plan["recent_changes"][0]["kind"] == "accepted_document"

    sup = as_(client, "supervisor").get("/app/teams/t1/plan/history").json()
    assert sup["total"] == 2 and all("text" not in e for e in sup["entries"])

    ex = as_(client, "examiner").get("/app/teams/t1/plan/history").json()
    assert all(e["document"] is None for e in ex["entries"])
    assert "Plan.txt" not in json.dumps(ex)


def test_the_history_route_rejects_unknown_fields_and_outsiders(client):
    assert client.get("/app/teams/t1/plan/history?field=nonsense").status_code == 404
    assert as_(client, "outsider").get("/app/teams/t1/plan/history").status_code == 403
    assert client.get("/app/teams/r1/plan/history").status_code == 403          # students: research teams are hidden
    assert as_(client, "supervisor").get("/app/teams/r1/plan/history").json()["available"] is False
