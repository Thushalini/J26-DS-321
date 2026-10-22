"""Tests for buddy/profile.py: the rules behind strengths and growth areas, the private profile and notes.

Runs on SQLite with stand-in app tables (the real column names the profile reads). Sign-in is replaced
by a fixed user. All names and messages are invented.
"""
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import buddy.profile as profile
from buddy.auth import current_user, get_conn
from component1.common.events import EventStore
from component1.decisions.tracker import endorse, object_to, propose
from component1.storage import ensure_c1_tables, save_new_events

USERS = {
    "u1": {"user_id": "u1", "email": "ayesha@example.com", "name": "Ayesha", "role": "student",
           "created_at": "2026-09-01T08:00:00Z"},
    "u2": {"user_id": "u2", "email": "bimal@example.com", "name": "Bimal", "role": "student",
           "created_at": "2026-09-02T08:00:00Z"},
    "u3": {"user_id": "u3", "email": "silva@example.com", "name": "Dr. Silva", "role": "supervisor",
           "created_at": "2026-08-01T08:00:00Z"},
}


def mix(**shares):
    full = dict.fromkeys(profile.STAGE_LABEL, 0.0)
    full.update(shares)
    return {"mix": full, "weighted_items": 20, "code_items": 14, "chat_items": 12}


NO_DECISIONS = {"proposed": 0, "with_reason": 0, "adopted": 0, "rejected": 0, "agreed": 0}
QUIET_CHAT = {"mine": 0, "team_average": 0.0}


def titles(items):
    return [i["title"] for i in items]


# ------------------------------------------------------------------ rules
def test_a_large_share_of_one_stage_is_a_strength_and_tiny_shares_are_growth_areas():
    s, g = profile.strengths_and_growth(mix(model_development=0.6, eda=0.3, data_cleaning=0.1), NO_DECISIONS, QUIET_CHAT)
    assert titles(s) == ["Strong in model development", "Strong in exploratory analysis (EDA)"]
    assert "60% of your work" in s[0]["evidence"] and s[0]["source"] == profile.BEHAVIOUR
    assert g[0]["title"].startswith("Little activity in feature engineering")


def test_too_little_work_is_not_judged():
    small = mix(model_development=1.0)
    small["weighted_items"] = 2
    assert profile.strengths_and_growth(small, NO_DECISIONS, QUIET_CHAT) == ([], [])


def test_decision_strengths_and_growth_areas():
    good = {"proposed": 4, "with_reason": 3, "adopted": 3, "rejected": 0, "agreed": 3}
    s, g = profile.strengths_and_growth(None, good, QUIET_CHAT)
    assert titles(s) == ["Drives planning decisions", "Explains choices", "Ideas the team adopts",
                         "Backs teammates' proposals"]
    assert g == []
    poor = {"proposed": 4, "with_reason": 0, "adopted": 0, "rejected": 2, "agreed": 0}
    s, g = profile.strengths_and_growth(None, poor, QUIET_CHAT)
    assert titles(g) == ["Give reasons when proposing", "Check ideas with the team before proposing"]
    assert "0 of 4 proposals had a reason" in g[0]["evidence"]


def test_chat_compared_with_the_team_average():
    s, _ = profile.strengths_and_growth(None, NO_DECISIONS, {"mine": 30, "team_average": 10.0})
    assert titles(s) == ["Active in the team chat"]
    _, g = profile.strengths_and_growth(None, NO_DECISIONS, {"mine": 2, "team_average": 10.0})
    assert titles(g) == ["Speak up more in the team chat"]


# ------------------------------------------------------------------- API
@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "app.db"

    def connect():
        return sqlite3.connect(str(path), check_same_thread=False)

    c = connect()
    c.executescript(
        """
        CREATE TABLE teams (team_id TEXT PRIMARY KEY, name TEXT, is_research INTEGER);
        CREATE TABLE team_members (team_id TEXT, user_id TEXT, team_role TEXT, member_code TEXT,
                                   git_identities TEXT, joined_at TEXT);
        CREATE TABLE chat_messages (msg_id TEXT, team_id TEXT, user_id TEXT, ts TEXT, text TEXT, reply_to TEXT);
        CREATE TABLE activities (activity_id TEXT, team_id TEXT, member_id TEXT, source TEXT, ts TEXT);
        INSERT INTO teams VALUES ('t1', 'Tourism Team', 0), ('r1', 'Research', 1);
        INSERT INTO team_members VALUES
            ('t1', 'u1', 'lead', 'M-1', '["ayesha"]', '2026-09-03T08:00:00Z'),
            ('t1', 'u2', 'member', 'M-2', NULL, '2026-09-03T09:00:00Z'),
            ('t1', 'u3', 'supervisor', NULL, NULL, '2026-09-04T09:00:00Z'),
            ('r1', 'u1', 'member', 'M-9', NULL, '2026-09-05T09:00:00Z');
        """
    )
    for i in range(12):
        c.execute("INSERT INTO chat_messages VALUES (?, 't1', 'u1', '2026-10-01T10:00:00Z', 'hello', NULL)", (f"a{i}",))
    c.execute("INSERT INTO chat_messages VALUES ('b1', 't1', 'u2', '2026-10-01T10:00:00Z', 'hi', NULL)")
    ensure_c1_tables(c)
    store = EventStore("t1")
    a = propose(store, "algorithm", "XGBoost", "M-1", "m1", "it handles tabular data well")
    endorse(store, store.state.decisions[a.decision_id], "M-2")
    b = propose(store, "evaluation_metric", "Accuracy", "M-1", "m2")
    object_to(store, store.state.decisions[b.decision_id], "M-2")
    propose(store, "dataset", "hotel records", "M-2", "m3")
    save_new_events(c, "t1", store.events)
    c.commit()
    c.close()
    monkeypatch.setattr(profile, "_notes_ready", False)
    return connect


def client_for(db, user_id):
    app = FastAPI()
    app.include_router(profile.router)

    def conn():
        c = db()
        try:
            yield c
        finally:
            c.close()

    app.dependency_overrides[get_conn] = conn
    app.dependency_overrides[current_user] = lambda: USERS[user_id]
    return TestClient(app)


def test_the_profile_shows_me_my_teams_and_my_record(db):
    p = client_for(db, "u1").get("/app/me/profile").json()
    assert p["about"] == {"user_id": "u1", "email": "ayesha@example.com", "name": "Ayesha", "role": "student",
                          "joined": "2026-09-01T08:00:00Z"}
    assert [t["name"] for t in p["teams"]] == ["Tourism Team"]             # the research team is left out
    assert p["teams"][0]["member_code"] == "M-1" and p["teams"][0]["git_identities"] == ["ayesha"]
    assert p["decisions"] == {"proposed": 2, "with_reason": 1, "adopted": 1, "rejected": 1, "agreed": 0}
    assert p["chat"] == {"mine": 12, "team_average": 1.0}
    assert "Active in the team chat" in titles(p["strengths"])
    assert p["has_activity"] and p["work_mix"] is None                     # no commits or labelled chat yet


def test_a_supervisor_sees_about_me_and_groups_only(db):
    p = client_for(db, "u3").get("/app/me/profile").json()
    assert p["about"]["role"] == "supervisor"
    assert [t["team_role"] for t in p["teams"]] == ["supervisor"]
    assert p["strengths"] == [] and p["growth"] == [] and not p["has_activity"]


def test_notes_are_saved_and_private_to_their_owner(db):
    me = client_for(db, "u1")
    saved = me.put("/app/me/notes", json={"strengths": " Good at SQL ", "improve": "Testing", "goals": "Ship v1"}).json()
    assert saved["strengths"] == "Good at SQL" and saved["goals"] == "Ship v1" and saved["updated_at"]
    assert me.get("/app/me/profile").json()["notes"]["improve"] == "Testing"
    assert client_for(db, "u2").get("/app/me/profile").json()["notes"]["strengths"] is None
    again = me.put("/app/me/notes", json={"strengths": "SQL and pandas"}).json()
    assert again["strengths"] == "SQL and pandas" and again["improve"] == ""


def test_notes_have_a_length_limit(db):
    assert client_for(db, "u1").put("/app/me/notes", json={"goals": "x" * 2001}).status_code == 422
