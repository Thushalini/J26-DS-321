"""Tests for saving to the database and surviving a restart (service + storage on SQLite).

A pretend language model and a small model trained on simulated data are used. All values
are invented. Each "restart" is a brand-new AgentService reading the same database file.
"""
import json
import sqlite3

from sklearn.ensemble import RandomForestClassifier

from component1.agent.policy import Buddy
from component1.agent.signals import GAP, Signal
from component1.common.schema import FieldName, FieldStatus
from component1.service import AgentService
from component1.state.features import FEATURE_ORDER
from component1.state.simulate import FEATURE_NAMES, generate_dataset
from component1.storage import list_agent_messages, load_events, timeline


def small_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


BUNDLE = small_bundle()

TOURIST = "I think we should predict tourist arrivals."
HOTEL = "Actually the target should be hotel occupancy."
XGB = "Let's use XGBoost, it handles tabular data well."
MEETING = "What time is the meeting tomorrow?"
SCRIPT = {
    TOURIST: [{"field": "target_variable", "value": "tourist arrivals", "span": "predict tourist arrivals", "hedged": True}],
    HOTEL: [{"field": "target_variable", "value": "hotel occupancy", "span": "hotel occupancy", "hedged": False}],
    XGB: [{"field": "algorithm", "value": "XGBoost", "span": "use XGBoost", "hedged": False}],
}
T0 = "2026-10-19T10:00:00Z"


def scripted(prompt):
    for text, items in SCRIPT.items():
        if text in prompt:
            return json.dumps({"items": items})
    return json.dumps({"items": []})


def connector(path):
    def connect():
        return sqlite3.connect(str(path))   # plain tuple rows, like the real Postgres connection
    return connect


def service(path, llm=scripted):
    return AgentService(bundle=BUNDLE, llm=llm, env_file=None, conn_factory=connector(path))


def raw(path):
    return connector(path)()


def test_a_reply_and_the_blueprint_changes_are_saved(tmp_path):
    db = tmp_path / "db.sqlite"
    (reply,) = service(db).handle_chat_message("t1", "m1", "M-1", TOURIST, T0)
    conn = raw(db)
    assert len(load_events(conn, "t1")) >= 2
    (saved,) = list_agent_messages(conn, "t1")
    assert saved["text"] == reply and saved["after_msg_id"] == "m1" and saved["ts"] == T0
    assert saved["kinds"] == ["saved"] and saved["details"]["keys"]


def test_after_a_restart_the_blueprint_is_back(tmp_path):
    db = tmp_path / "db.sqlite"
    first = service(db)
    first.handle_chat_message("t1", "m1", "M-1", TOURIST, T0)
    first.handle_chat_message("t1", "m2", "M-2", HOTEL, "2026-10-19T10:00:20Z")
    second = service(db)                                    # a restart
    second.handle_chat_message("t1", "m3", "M-3", MEETING, "2026-10-19T10:00:40Z")
    state = second._agents["t1"].store.state
    assert state.fields[FieldName.TARGET_VARIABLE].status == FieldStatus.CONFLICTING
    assert state.fields[FieldName.TARGET_VARIABLE].value == "hotel occupancy"


def test_after_a_restart_buddy_does_not_repeat_the_conflict_question(tmp_path):
    db = tmp_path / "db.sqlite"
    first = service(db)
    first.handle_chat_message("t1", "m1", "M-1", TOURIST, T0)
    (asked,) = first.handle_chat_message("t1", "m2", "M-2", HOTEL, "2026-10-19T10:00:20Z")
    assert "different answers" in asked
    second = service(db)
    assert second.handle_chat_message("t1", "m3", "M-3", MEETING, "2026-10-19T10:00:40Z") == []


def test_a_message_already_handled_is_not_handled_again_after_a_restart(tmp_path):
    db = tmp_path / "db.sqlite"
    service(db).handle_chat_message("t1", "m1", "M-1", TOURIST, T0)
    before = len(load_events(raw(db), "t1"))
    assert service(db).handle_chat_message("t1", "m1", "M-1", TOURIST, T0) == []
    assert len(load_events(raw(db), "t1")) == before
    assert len(list_agent_messages(raw(db), "t1")) == 1


def test_a_question_asked_on_the_timer_is_saved_too(tmp_path):
    db = tmp_path / "db.sqlite"
    svc = service(db)
    svc.handle_chat_message("t1", "m1", "M-1", XGB, T0)
    (reply,) = svc.tick("t1", "2026-10-19T10:03:20Z")
    saved = list_agent_messages(raw(db), "t1")[-1]
    assert saved["text"] == reply and saved["after_msg_id"] is None
    assert saved["kinds"] == ["gap"] and saved["ts"] == "2026-10-19T10:03:20Z"


def test_the_screen_gets_one_timeline_of_members_and_buddy(tmp_path):
    db = tmp_path / "db.sqlite"
    conn = raw(db)
    conn.executescript(
        "CREATE TABLE users (user_id TEXT PRIMARY KEY, name TEXT);"
        "CREATE TABLE team_members (team_id TEXT, user_id TEXT, team_role TEXT, member_code TEXT);"
        "CREATE TABLE chat_messages (msg_id TEXT PRIMARY KEY, team_id TEXT, user_id TEXT, ts TEXT, text TEXT, reply_to TEXT);"
        "INSERT INTO users VALUES ('u1', 'Ayesha');"
        "INSERT INTO team_members VALUES ('t1', 'u1', 'lead', 'M-1');"
    )
    conn.execute("INSERT INTO chat_messages VALUES ('m1', 't1', 'u1', ?, ?, NULL)", (T0, TOURIST))
    conn.commit()
    service(db).handle_chat_message("t1", "m1", "M-1", TOURIST, T0)
    items = timeline(raw(db), "t1")
    assert [i["sender_type"] for i in items] == ["member", "assistant"]
    assert items[1]["text"].startswith("Noted:")


def test_a_database_that_is_down_never_breaks_the_chat(tmp_path):
    def broken():
        raise ConnectionError("the database is down")

    svc = AgentService(bundle=BUNDLE, llm=scripted, env_file=None, conn_factory=broken)
    assert svc.handle_chat_message("t1", "m1", "M-1", TOURIST, T0) == []
    assert svc.tick("t1", T0) == []


def test_buddys_own_messages_leave_no_trace_in_the_database(tmp_path):
    db = tmp_path / "db.sqlite"
    svc = service(db)
    assert svc.handle_chat_message("t1", "m1", None, TOURIST, T0, sender_type="assistant") == []
    assert len(svc.handle_chat_message("t1", "m1", "M-1", TOURIST, T0)) == 1   # the id was not used up


def test_buddy_does_not_repeat_what_it_was_told_it_already_said():
    buddy = Buddy()
    buddy.restore([["gap", "problem", "", ""]], at=100.0, asked=True)
    assert buddy.step([Signal(GAP, "problem")], now=200.0) == []                 # already asked
    assert buddy.step([Signal(GAP, "dataset")], now=130.0) == []                 # too soon after that question