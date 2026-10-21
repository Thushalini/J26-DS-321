"""The team endpoints on a throwaway SQLite database (never the shared one)."""
import base64

import joblib
import pytest


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    import component2.config as c2config
    import component2.db as c2db
    from fastapi.testclient import TestClient

    if c2config.DATABASE_URL:
        pytest.skip("Refusing to run against a real database: DATABASE_URL is set.")

    from component4.api import router as api_router
    from component4.api import service
    from component4.common import config
    from component4.m2_prediction.data import split
    from component4.m2_prediction.train import train_and_evaluate
    from component4.synthetic import generator

    folder = tmp_path_factory.mktemp("team")
    c2db.DB_PATH = folder / "test.db"
    frame = generator.generate(500)
    _, model = train_and_evaluate(frame, search=False)
    parts = split(frame)
    model_path = folder / "model.joblib"
    joblib.dump({"model": model, "version": "test", "background": parts["train"][0].sample(20, random_state=0),
                 "medians": parts["train"][0].median().to_dict(), "demo_project": parts["test"][0].iloc[0].to_dict()},
                model_path)
    original = config.MODEL_PATH
    config.MODEL_PATH = model_path
    service._bundle.clear()

    import main
    client = TestClient(main.app)
    yield client, api_router
    config.MODEL_PATH = original
    service._bundle.clear()


def account(client, email, role="student"):
    r = client.post("/app/auth/register", json={"email": email, "name": email.split("@")[0], "password": "password123", "role": role})
    return {"Authorization": "Bearer " + r.json()["token"]}, r.json()["user"]["user_id"]


@pytest.fixture(scope="module")
def team(app):
    client, _ = app
    lead, lead_id = account(client, "lead@t.io")
    member, member_id = account(client, "member@t.io")
    sup, _ = account(client, "sup@t.io", "supervisor")
    t = client.post("/app/teams", json={"name": "Team"}, headers=lead).json()
    client.post("/app/teams/join", json={"join_code": t["join_code"]}, headers=member)
    code = client.get("/app/auth/me", headers=sup).json()["supervisor_code"]
    client.post(f"/app/teams/{t['team_id']}/supervisors", json={"supervisor_code": code}, headers=lead)
    return {"id": t["team_id"], "lead": lead, "lead_id": lead_id, "member_id": member_id, "sup": sup}


def test_an_empty_group_gets_no_number(app, team):
    client, _ = app
    body = client.get(f"/c4/teams/{team['id']}/prediction", headers=team["lead"]).json()
    assert body["enough_data"] is False and body["probability"] is None
    assert body["signals_from_team"] + len(body["imputed_features"]) == 15
    assert all("prediction" not in w["sources"] for w in body["weak_stages"])


RUNS = [("HIGH", 0.86, "overfitting"), ("HIGH", 0.81, "overfitting"), ("LOW", 0.2, None)]
FILES = ["README.md", "requirements.txt", "tests/test_model.py", ".github/workflows/ci.yml", "notebooks/eda.ipynb"]


@pytest.fixture(scope="module")
def busy_team(app):
    import component2.db as c2db
    from component3.api import store as c3store

    client, _ = app
    lead, _ = account(client, "busy@t.io")
    t = client.post("/app/teams", json={"name": "Busy"}, headers=lead).json()
    conn = c2db.connect()
    assert not conn.is_postgres
    c3store.ensure_schema(conn)
    for index, (level, severity, cause) in enumerate(RUNS):
        c3store.save_alert(conn, t["team_id"], {"run_id": f"run{index}", "severity_level": level, "severity_score": severity,
                                                "anomaly_score": severity, "recommended_cause": cause,
                                                "root_cause_probabilities": {}})
    c2db.save_activities(conn, [{"activity_id": "commit:busy:1", "team_id": t["team_id"], "source": "commit",
                                 "member_id": None, "ts": "2026-10-01T00:00:00Z", "text": "start",
                                 "meta": {"files": FILES}}])
    return {"id": t["team_id"], "lead": lead}


def test_a_group_with_enough_data_gets_a_prediction(app, busy_team):
    client, _ = app
    body = client.get(f"/c4/teams/{busy_team['id']}/prediction", headers=busy_team["lead"]).json()
    assert body["enough_data"] is True and 0 <= body["probability"] <= 100
    assert body["signals_from_team"] == 7 and body["readiness"]["score"] > 0
    assert body["history"] == [body["probability"]]


def test_stages_are_not_called_lagging_on_too_little_work(app, busy_team):
    client, _ = app
    body = client.get(f"/c4/teams/{busy_team['id']}/prediction", headers=busy_team["lead"]).json()
    assert body["lagging_stages"] == []
    assert all("behavior" not in w["sources"] for w in body["weak_stages"])


def test_recurring_issues_are_counted(app, busy_team):
    client, _ = app
    body = client.get(f"/c4/teams/{busy_team['id']}/prediction", headers=busy_team["lead"]).json()
    assert any("overfitting (2 issues)" in r["reason"] for r in body["recommendations"])


def test_recommendations_never_cite_unmeasured_signals(app, busy_team):
    client, _ = app
    from component4.common import config
    body = client.get(f"/c4/teams/{busy_team['id']}/prediction", headers=busy_team["lead"]).json()
    unmeasured = [config.FEATURE_LABELS[name] for name in body["imputed_features"]]
    for recommendation in body["recommendations"]:
        assert not any(label in recommendation["reason"] for label in unmeasured)
    assert {f["id"] for f in body["factors"] if not f["measured"]} == set(body["imputed_features"])


def test_supervisor_sees_the_prediction(app, team):
    client, _ = app
    assert client.get(f"/c4/teams/{team['id']}/prediction", headers=team["sup"]).status_code == 200


def test_strangers_cannot_see_it(app, team):
    client, _ = app
    stranger, _ = account(client, "stranger@t.io")
    assert client.get(f"/c4/teams/{team['id']}/prediction", headers=stranger).status_code == 403


def _sheet(member_id, text):
    return {"member_user_id": member_id, "content_base64": base64.b64encode(text.encode()).decode()}


STRONG = "IT3071 Machine Learning and Optimization Methods 4 1 A+\nIT3011 Theory and Practices in Statistical Modelling 4 1 A"
WEAK = "IT3071 Machine Learning and Optimization Methods 4 1 C\nIT3011 Theory and Practices in Statistical Modelling 4 1 C"


@pytest.fixture
def text_sheets(app, monkeypatch):
    client, api_router = app
    from component4.m6_improvement.competency.sheets import parse_text
    monkeypatch.setattr(api_router, "parse_sheet", lambda data: parse_text(data.decode()))
    return client


def test_prediction_lists_weak_stages_with_their_sources(app, team):
    client, _ = app
    body = client.get(f"/c4/teams/{team['id']}/prediction", headers=team["lead"]).json()
    assert all({"stage", "label", "sources"} <= set(w) for w in body["weak_stages"])


def test_anyone_in_the_group_can_save_sheets_for_anyone(text_sheets, team):
    client = text_sheets
    client.post(f"/c4/teams/{team['id']}/sheets", headers=team["lead"], json=_sheet(team["member_id"], STRONG))
    listed = client.post(f"/c4/teams/{team['id']}/sheets", headers=team["sup"], json=_sheet(team["lead_id"], WEAK)).json()
    assert {m["user_id"]: m["saved"] for m in listed} == {team["lead_id"]: True, team["member_id"]: True}
    assert "grade" not in str(listed) and "modules" not in str(listed)


def test_saved_sheets_give_suggestions_without_marks(text_sheets, team):
    client = text_sheets
    client.post(f"/c4/teams/{team['id']}/sheets", headers=team["lead"], json=_sheet(team["member_id"], STRONG))
    client.post(f"/c4/teams/{team['id']}/sheets", headers=team["lead"], json=_sheet(team["lead_id"], WEAK))
    body = client.post(f"/c4/teams/{team['id']}/helpers", headers=team["lead"], json={"stages": ["model_development"]}).json()
    (suggestion,) = body["suggestions"]
    assert suggestion["member_id"] == team["member_id"] and "A+" not in str(body)


def test_removing_a_sheet(text_sheets, team):
    client = text_sheets
    client.post(f"/c4/teams/{team['id']}/sheets", headers=team["lead"], json=_sheet(team["member_id"], STRONG))
    listed = client.delete(f"/c4/teams/{team['id']}/sheets/{team['member_id']}", headers=team["lead"]).json()
    assert {m["user_id"]: m["saved"] for m in listed}[team["member_id"]] is False


def test_strangers_cannot_upload(app, team):
    client, _ = app
    stranger, _ = account(client, "stranger2@t.io")
    r = client.post(f"/c4/teams/{team['id']}/sheets", headers=stranger, json=_sheet(team["lead_id"], "x"))
    assert r.status_code == 403


def test_sheets_must_belong_to_group_members(app, team):
    client, _ = app
    r = client.post(f"/c4/teams/{team['id']}/sheets", headers=team["lead"], json=_sheet("someone-else", "x"))
    assert r.status_code == 422


def test_helpers_need_at_least_one_saved_sheet(app):
    client, _ = app
    lead, _ = account(client, "lead2@t.io")
    t = client.post("/app/teams", json={"name": "Empty"}, headers=lead).json()
    assert client.post(f"/c4/teams/{t['team_id']}/helpers", headers=lead, json={"stages": ["eda"]}).status_code == 422
