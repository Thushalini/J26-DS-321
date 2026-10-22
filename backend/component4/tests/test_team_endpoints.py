"""The team endpoints on a throwaway SQLite database (never the shared one)."""
import base64
from datetime import datetime, timezone

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
                                 "member_id": None, "ts": datetime.now(timezone.utc).isoformat(), "text": "start",
                                 "meta": {"files": FILES}}])
    return {"id": t["team_id"], "lead": lead}


def test_a_group_with_enough_data_gets_a_prediction(app, busy_team):
    client, _ = app
    body = client.get(f"/c4/teams/{busy_team['id']}/prediction", headers=busy_team["lead"]).json()
    assert body["enough_data"] is True and 0 <= body["probability"] <= 100
    assert body["signals_from_team"] == 8 and body["readiness"]["score"] > 0
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


def test_todays_number_comes_with_a_range(app, busy_team):
    client, _ = app
    body = client.get(f"/c4/teams/{busy_team['id']}/prediction", headers=busy_team["lead"]).json()
    spread = body["probability_range"]
    assert spread["low"] <= body["probability"] <= spread["high"] and spread["high"] > spread["low"]


def test_supervisor_outlook_suggests_a_dot_per_group(app, team):
    import component2.db as c2db
    client, _ = app
    conn = c2db.connect()
    conn.execute("INSERT INTO teams (team_id, name, is_research, created_at) VALUES (?, ?, 1, ?) ON CONFLICT (team_id) DO NOTHING",
                 ("research__outlook", "research repo", "2026-10-22T00:00:00Z"))
    conn.commit()
    rows = client.get("/c4/supervisor/outlook", headers=team["sup"]).json()
    ids = [r["team_id"] for r in rows]
    assert ids[0] == team["id"] and rows[0]["status"] is None
    assert "research__outlook" in ids and len(ids) == len(set(ids))


def test_only_supervisors_get_an_outlook(app, team):
    client, _ = app
    assert client.get("/c4/supervisor/outlook", headers=team["lead"]).status_code == 403


def test_status_rules():
    from component4.api.outlook import status
    assert status(80, {}) == "good" and status(60, {}) == "fine" and status(40, {}) == "attention"
    assert status(90, {"open_high_issues": 2}) == "attention"


def test_the_group_records_milestone_results_and_the_supervisor_only_reads(app, team):
    client, _ = app
    url = f"/c4/teams/{team['id']}/milestone-results"
    entry = {"milestone": "PP1", "milestone_date": "2026-12-15", "result": "passed_with_issues"}
    listed = client.post(url, headers=team["lead"], json=entry).json()
    assert [(r["milestone"], r["date"], r["result"]) for r in listed] == [("PP1", "2026-12-15", "passed_with_issues")]
    assert client.get(url, headers=team["sup"]).json() == listed
    assert client.post(url, headers=team["sup"], json=entry).status_code == 403
    assert client.post(url, headers=team["lead"], json={**entry, "result": "great"}).status_code == 422
    assert client.delete(f"{url}/{listed[0]['result_id']}", headers=team["lead"]).json() == []


def test_results_are_paired_with_the_last_prediction_before_the_milestone(app, busy_team):
    import component2.db as c2db
    from component4.m6_improvement import milestone_results
    client, _ = app
    client.get(f"/c4/teams/{busy_team['id']}/prediction", headers=busy_team["lead"])
    client.post(f"/c4/teams/{busy_team['id']}/milestone-results", headers=busy_team["lead"],
                json={"milestone": "Viva", "milestone_date": "2099-01-01", "result": "passed_well"})
    paired = [p for p in milestone_results.paired_with_predictions(c2db.connect()) if p["team_id"] == busy_team["id"]]
    assert paired and paired[0]["probability_before"] is not None


def test_readiness_reads_key_repo_files_when_the_group_has_a_github_link(app, monkeypatch):
    from component4.m4_readiness import repo_scan
    client, _ = app
    files = {"README.md": "## Setup\npip install -r requirements.txt\n" + "word " * 120,
             "tests/test_a.py": "def test_a():\n    pass\n" * 4, "requirements.txt": "pandas==2.2\n"}
    monkeypatch.setattr(repo_scan, "github_fetcher", lambda owner, repo: ({p: len(t) for p, t in files.items()}, files.get))
    lead, _ = account(client, "repo@t.io")
    t = client.post("/app/teams", json={"name": "Repo", "repo_url": "https://github.com/team/project"}, headers=lead).json()
    body = client.get(f"/c4/teams/{t['team_id']}/prediction", headers=lead).json()
    assert body["readiness"]["basis"] == "contents"
    pipeline = next(a for a in body["readiness"]["areas"] if a["id"] == "pipeline")
    assert "automated tests (4 tests)" in pipeline["reason"]


def test_component1_timeline_and_milestones_give_timeline_adherence(app):
    import component2.db as c2db
    from component1.common.events import EventStore
    from component1.common.schema import FieldName, FieldStatus
    from component1.storage import ensure_c1_tables, save_new_events
    from component1.timeline.milestones import record_milestone
    from component4.m1_feature_integration.team_sources import team_project
    client, _ = app
    lead, _ = account(client, "timeline@t.io")
    t = client.post("/app/teams", json={"name": "Timeline"}, headers=lead).json()
    store = EventStore(t["team_id"])
    store.update_field(FieldName.TIMELINE, "deadline 2025-06-30", FieldStatus.CONFIRMED, 0.9, None, "test")
    record_milestone(store, "milestone: Demo, 2025-03-01", "demo is on 1 March", "M-1", "m1")
    store.update_scores(actor="test", feasibility={"model_feasibility": "pass", "compute_fits_model": "warning"})
    conn = c2db.connect()
    ensure_c1_tables(conn)
    save_new_events(conn, t["team_id"], store.events)
    project = team_project(conn, t["team_id"], allow_network=False)
    own = [m for m in project["milestones"] if m["source"] == "group"]
    assert [m["name"] for m in own] == ["Demo", "Final deadline"]
    assert all(m["due"] and m["met"] is False for m in own)
    assert project["features"]["timeline_adherence"] == 0.0
    assert project["features"]["novelty_score"] is None
    assert project["features"]["feasibility_score"] == 0.75
    body = client.get(f"/c4/teams/{t['team_id']}/prediction", headers=lead).json()
    assert [m["name"] for m in body["milestones"]][:2] == ["Demo", "Final deadline"]


def test_every_group_gets_the_shared_schedule_from_its_start(app, team):
    from component4.common import config
    client, _ = app
    body = client.get(f"/c4/teams/{team['id']}/prediction", headers=team["lead"]).json()
    shared = [m for m in body["milestones"] if m["source"] == "schedule"]
    assert [m["name"] for m in shared] == [m["name"] for m in config.COHORT_SCHEDULE["milestones"]]
    assert not any(m["due"] for m in shared)
    basis = body["timeline_basis"]
    assert basis["start_from"] == "group" and basis["schedule"] == config.COHORT_SCHEDULE["name"]


def test_research_teams_skip_the_shared_schedule(app):
    import component2.db as c2db
    from component4.m1_feature_integration.team_sources import team_project
    conn = c2db.connect()
    conn.execute("INSERT INTO teams (team_id, name, is_research, created_at) VALUES (?, ?, 1, ?) ON CONFLICT (team_id) DO NOTHING",
                 ("research__schedule", "research repo", "2020-01-01T00:00:00Z"))
    conn.commit()
    project = team_project(conn, "research__schedule", allow_network=False)
    assert project["milestones"] == [] and project["timeline_basis"]["schedule"] is None
