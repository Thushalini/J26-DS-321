"""Tests for component1/storage.py on SQLite (the shared app falls back to SQLite too).

The users / team_members / chat_messages tables below are STAND-INS built from the real
column list. All values are invented. Plain tuple rows are used, like the real Postgres connection.
"""
import sqlite3

from component1.common.events import EventStore
from component1.common.schema import FieldName, FieldStatus, Provenance, SourceType
from component1.storage import (
    ensure_c1_tables,
    list_agent_messages,
    load_events,
    mark_processed,
    save_agent_message,
    save_new_events,
    timeline,
)


def connect(path):
    return sqlite3.connect(str(path))   # plain tuple rows, like the real Postgres connection


def shared_tables(conn):
    conn.executescript(
        """
        CREATE TABLE users (user_id TEXT PRIMARY KEY, name TEXT, email TEXT);
        CREATE TABLE team_members (team_id TEXT, user_id TEXT, team_role TEXT, member_code TEXT);
        CREATE TABLE chat_messages (msg_id TEXT PRIMARY KEY, team_id TEXT, user_id TEXT, ts TEXT, text TEXT, reply_to TEXT);
        INSERT INTO users VALUES ('u1', 'Ayesha', 'a@x'), ('u2', 'Bimal', 'b@x');
        INSERT INTO team_members VALUES ('t1', 'u1', 'lead', 'M-1'), ('t1', 'u2', 'member', 'M-2');
        """
    )
    conn.commit()


def say(conn, msg_id, user_id, ts, text, team_id="t1", reply_to=None):
    conn.execute("INSERT INTO chat_messages VALUES (?, ?, ?, ?, ?, ?)", (msg_id, team_id, user_id, ts, text, reply_to))
    conn.commit()


def prov(text="use XGBoost", msg="M1"):
    return Provenance(source_type=SourceType.CHAT, source_id=msg, speaker="M-1", text_span=text)


def store_with_events():
    store = EventStore("t1")
    store.update_field(FieldName.ALGORITHM, "XGBoost", FieldStatus.KNOWN, 0.5, prov(), "extractor")
    store.update_field(FieldName.ALGORITHM, "XGBoost", FieldStatus.CONFIRMED, 0.9, None, "state_classifier")
    store.update_scores("scorer", novelty_score=0.4)
    return store


# --------------------------------------------------------------- set-up
def test_the_tables_can_be_created_any_number_of_times(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    ensure_c1_tables(conn)
    ensure_c1_tables(conn)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"c1_events", "c1_agent_messages", "c1_processed"} <= names


# --------------------------------------------------------------- events
def test_events_are_saved_once_and_only_new_ones_are_added(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    ensure_c1_tables(conn)
    store = store_with_events()
    assert save_new_events(conn, "t1", store.events) == 3
    assert save_new_events(conn, "t1", store.events) == 0
    store.update_field(FieldName.DATASET, "tourism data", FieldStatus.KNOWN, 0.5, prov("tourism data"), "extractor")
    assert save_new_events(conn, "t1", store.events) == 1


def test_a_blueprint_rebuilt_from_saved_events_is_identical(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    ensure_c1_tables(conn)
    store = store_with_events()
    save_new_events(conn, "t1", store.events)
    rebuilt = EventStore.from_events("t1", load_events(conn, "t1"))
    assert rebuilt.state.model_dump(mode="json") == store.state.model_dump(mode="json")
    assert rebuilt.state.version == 3


def test_each_teams_events_are_kept_apart(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    ensure_c1_tables(conn)
    save_new_events(conn, "t1", store_with_events().events)
    assert load_events(conn, "t2") == []


# ------------------------------------------------------ processed ids
def test_a_message_is_marked_processed_only_once_per_team(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    ensure_c1_tables(conn)
    assert mark_processed(conn, "t1", "m1") is True
    assert mark_processed(conn, "t1", "m1") is False
    assert mark_processed(conn, "t2", "m1") is True


# ----------------------------------------------------- Buddy's messages
def test_buddys_messages_are_saved_with_their_reasons_and_listed_oldest_first(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    ensure_c1_tables(conn)
    save_agent_message(conn, "t1", "second", "2026-10-19T10:05:00Z", "m2", ["gap"], {"keys": [["gap", "problem", "", ""]]})
    save_agent_message(conn, "t1", "first", "2026-10-19T10:00:01Z", "m1", ["saved"], {"asked": False})
    msgs = list_agent_messages(conn, "t1")
    assert [m["text"] for m in msgs] == ["first", "second"]
    assert msgs[0]["kinds"] == ["saved"] and msgs[0]["details"] == {"asked": False}
    assert msgs[1]["details"]["keys"] == [["gap", "problem", "", ""]]
    assert msgs[0]["status"] == "open" and msgs[0]["sender_type"] == "assistant"


# ----------------------------------------------------------- timeline
def test_the_timeline_mixes_members_and_buddy_in_time_order(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    shared_tables(conn)
    ensure_c1_tables(conn)
    say(conn, "m1", "u1", "2026-10-19T10:00:00Z", "I think we should predict tourist arrivals.")
    say(conn, "m2", "u2", "2026-10-19T10:00:20Z", "Let's use XGBoost.")
    save_agent_message(conn, "t1", "Noted: target variable ...", "2026-10-19T10:00:00Z", "m1", ["saved"])
    save_agent_message(conn, "t1", "Noted: algorithm ...", "2026-10-19T10:00:20Z", "m2", ["saved"])
    items = timeline(conn, "t1")
    assert [i["sender_type"] for i in items] == ["member", "assistant", "member", "assistant"]
    assert items[0]["sender_name"] == "Ayesha" and items[0]["member_code"] == "M-1"
    assert items[1]["sender_name"] == "Buddy" and items[1]["after_msg_id"] == "m1"


def test_at_the_same_second_the_members_message_comes_first(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    shared_tables(conn)
    ensure_c1_tables(conn)
    save_agent_message(conn, "t1", "reply", "2026-10-19T10:00:00Z", "m1", ["saved"])   # saved before the member row
    say(conn, "m1", "u1", "2026-10-19T10:00:00Z", "hello")
    assert [i["sender_type"] for i in timeline(conn, "t1")] == ["member", "assistant"]


def test_the_timeline_returns_the_latest_items_and_can_page_back_and_refresh(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    shared_tables(conn)
    ensure_c1_tables(conn)
    for i in range(5):
        say(conn, f"m{i}", "u1", f"2026-10-19T10:0{i}:00Z", f"message {i}")
    save_agent_message(conn, "t1", "buddy", "2026-10-19T10:02:30Z", "m2", ["saved"])
    latest = timeline(conn, "t1", limit=3)
    assert [i["text"] for i in latest] == ["buddy", "message 3", "message 4"]
    older = timeline(conn, "t1", limit=3, before_ts=latest[0]["ts"])
    assert [i["text"] for i in older] == ["message 0", "message 1", "message 2"]
    newer = timeline(conn, "t1", after_ts="2026-10-19T10:03:00Z")
    assert [i["text"] for i in newer] == ["message 4"]


def test_a_member_message_from_an_unknown_user_still_shows(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    shared_tables(conn)
    ensure_c1_tables(conn)
    say(conn, "m1", "ghost", "2026-10-19T10:00:00Z", "who am I")
    (item,) = timeline(conn, "t1")
    assert item["sender_name"] is None and item["text"] == "who am I"


def test_other_teams_messages_never_appear(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    shared_tables(conn)
    ensure_c1_tables(conn)
    say(conn, "m1", "u1", "2026-10-19T10:00:00Z", "ours")
    say(conn, "m9", "u1", "2026-10-19T10:00:01Z", "theirs", team_id="t2")
    save_agent_message(conn, "t2", "theirs too", "2026-10-19T10:00:02Z")
    assert [i["text"] for i in timeline(conn, "t1")] == ["ours"]


def test_component_1_never_changes_the_shared_chat_table(tmp_path):
    conn = connect(tmp_path / "db.sqlite")
    shared_tables(conn)
    say(conn, "m1", "u1", "2026-10-19T10:00:00Z", "hello")
    before = [tuple(r) for r in conn.execute("SELECT * FROM chat_messages")]
    ensure_c1_tables(conn)
    save_agent_message(conn, "t1", "x", "2026-10-19T10:00:01Z")
    mark_processed(conn, "t1", "m1")
    assert [tuple(r) for r in conn.execute("SELECT * FROM chat_messages")] == before