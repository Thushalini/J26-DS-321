"""Tests for the app hook (component1/hook.py), the chat merge and the old-blueprint mirror.

SQLite with plain tuple rows stands in for the shared database. The `blueprints` table below
is a stand-in built from the real column list. All values are invented.
"""
import json
import sqlite3
import time

from sklearn.ensemble import RandomForestClassifier

from component1 import hook
from component1.service import AgentService
from component1.state.features import FEATURE_ORDER
from component1.state.simulate import FEATURE_NAMES, generate_dataset
from component1.storage import (
    ensure_c1_tables,
    list_agent_messages,
    merge_agent_messages,
    save_agent_message,
)

T0 = "2026-10-19T10:00:00Z"
TOURIST = "I think we should predict tourist arrivals."
XGB = "Let's use XGBoost, it handles tabular data well."
SCRIPT = {
    TOURIST: [{"field": "target_variable", "value": "tourist arrivals", "span": "predict tourist arrivals", "hedged": True}],
    XGB: [{"field": "algorithm", "value": "XGBoost", "span": "use XGBoost", "hedged": False}],
}


def small_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


BUNDLE = small_bundle()


def scripted(prompt):
    for text, items in SCRIPT.items():
        if text in prompt:
            return json.dumps({"items": items})
    return json.dumps({"items": []})


def connector(path):
    def connect():
        return sqlite3.connect(str(path))
    return connect


def service(path):
    return AgentService(bundle=BUNDLE, llm=scripted, env_file=None, conn_factory=connector(path))


def with_blueprints(path, row=None):
    conn = connector(path)()
    conn.execute(
        "CREATE TABLE blueprints (team_id TEXT PRIMARY KEY, target TEXT, algorithm TEXT, metric TEXT, "
        "notes TEXT, updated_by TEXT, updated_at TEXT NOT NULL)"
    )
    if row:
        conn.execute("INSERT INTO blueprints (team_id, target, algorithm, metric, notes, updated_by, updated_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?)", row)
    conn.commit()
    conn.close()


def blueprint_row(path, team_id="t1"):
    conn = connector(path)()
    cur = conn.execute("SELECT target, algorithm, metric, notes, updated_by FROM blueprints WHERE team_id = ?", (team_id,))
    row = cur.fetchone()
    conn.close()
    return None if row is None else dict(zip(["target", "algorithm", "metric", "notes", "updated_by"], row))


def member(msg_id, ts, text, sender="Ayesha"):
    return {"msg_id": msg_id, "sender": sender, "member_code": "M-1", "ts": ts, "text": text,
            "reply_to": None, "mine": False, "team_role": "member"}


# ------------------------------------------------------------ chat merge
def test_buddys_messages_join_the_listing_in_the_same_shape(tmp_path):
    conn = connector(tmp_path / "db.sqlite")()
    ensure_c1_tables(conn)
    save_agent_message(conn, "t1", "Noted: x", "2026-10-19T10:00:00Z", "m1", ["saved"])
    out = merge_agent_messages(conn, "t1", [member("m1", "2026-10-19T10:00:00Z", "hello")])
    assert [m["sender_type"] for m in out] == ["member", "assistant"]
    buddy = out[1]
    assert buddy["sender"] == "Buddy" and buddy["reply_to"] == "m1" and buddy["mine"] is False
    assert buddy["team_role"] == "assistant" and buddy["kinds"] == ["saved"] and buddy["member_code"] is None
    assert set(member("x", "t", "t")) <= set(buddy)          # every key the screen already uses is there


def test_the_listing_is_in_time_order_with_members_first_on_a_tie(tmp_path):
    conn = connector(tmp_path / "db.sqlite")()
    ensure_c1_tables(conn)
    save_agent_message(conn, "t1", "reply to m2", "2026-10-19T10:00:20Z", "m2", ["saved"])
    save_agent_message(conn, "t1", "reply to m1", "2026-10-19T10:00:00Z", "m1", ["saved"])
    out = merge_agent_messages(conn, "t1", [member("m1", "2026-10-19T10:00:00Z", "a"), member("m2", "2026-10-19T10:00:20Z", "b")])
    assert [m["text"] for m in out] == ["a", "reply to m1", "b", "reply to m2"]


def test_buddy_messages_older_than_the_visible_window_are_left_out(tmp_path):
    conn = connector(tmp_path / "db.sqlite")()
    ensure_c1_tables(conn)
    save_agent_message(conn, "t1", "very old", "2026-10-19T09:00:00Z", None, ["gap"])
    save_agent_message(conn, "t1", "recent", "2026-10-19T10:00:05Z", "m1", ["saved"])
    window = [member("m1", "2026-10-19T10:00:00Z", "a"), member("m2", "2026-10-19T10:00:20Z", "b")]
    out = merge_agent_messages(conn, "t1", window, limit=2)        # the window is full
    assert "very old" not in [m["text"] for m in out]


def test_other_teams_buddy_messages_never_appear(tmp_path):
    conn = connector(tmp_path / "db.sqlite")()
    ensure_c1_tables(conn)
    save_agent_message(conn, "t2", "not ours", T0, None, ["gap"])
    assert merge_agent_messages(conn, "t1", [member("m1", T0, "a")]) == [dict(member("m1", T0, "a"), sender_type="member")]


def test_merge_chat_creates_the_tables_if_nobody_has_chatted_yet(tmp_path):
    hook._tables_ready = False
    conn = connector(tmp_path / "db.sqlite")()
    messages = [member("m1", T0, "first message ever")]
    out = hook.merge_chat(conn, "t1", messages)
    assert [m["text"] for m in out] == ["first message ever"]


def test_merge_chat_returns_the_original_list_if_anything_goes_wrong():
    class Broken:
        def execute(self, *args, **kwargs):
            raise RuntimeError("database trouble")

    hook._tables_ready = False
    messages = [member("m1", T0, "a")]
    assert hook.merge_chat(Broken(), "t1", messages) == messages


# ------------------------------------------------------------- background
class FakeTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args):
        self.tasks.append((func, args))


def test_the_hook_schedules_one_background_task_and_it_stores_buddys_reply(tmp_path):
    db = tmp_path / "db.sqlite"
    hook._service = service(db)
    try:
        tasks = FakeTasks()
        hook.run_in_background(tasks, "t1", "m1", "M-1", "u1", T0, TOURIST)
        assert len(tasks.tasks) == 1
        func, args = tasks.tasks[0]
        func(*args)                                           # what FastAPI does after the response
        (saved,) = list_agent_messages(connector(db)(), "t1")
        assert saved["text"].startswith("Noted:")
    finally:
        hook._service = None


def test_scheduling_never_raises_even_if_the_task_runner_fails():
    class Broken:
        def add_task(self, *args):
            raise RuntimeError("no task runner")

    hook.run_in_background(Broken(), "t1", "m1", "M-1", "u1", T0, TOURIST)   # must not raise


# ---------------------------------------------------- old blueprint mirror
def test_settled_values_are_copied_into_the_old_blueprint_row(tmp_path):
    db = tmp_path / "db.sqlite"
    with_blueprints(db)
    service(db).handle_chat_message("t1", "m1", "M-1", XGB, T0)
    row = blueprint_row(db)
    assert row["algorithm"] == "XGBoost" and row["updated_by"] == "component1"
    assert row["target"] is None


def test_an_unsettled_value_is_not_copied(tmp_path):
    db = tmp_path / "db.sqlite"
    with_blueprints(db)
    svc = service(db)
    svc.handle_chat_message("t1", "m1", "M-1", TOURIST, T0)                # hedged, so only "uncertain"
    assert blueprint_row(db) is None                                        # nothing settled, nothing written
    svc.handle_chat_message("t1", "m2", "M-1", XGB, "2026-10-19T10:00:20Z")
    assert blueprint_row(db)["target"] is None


def test_a_value_typed_by_hand_is_never_erased_and_notes_are_kept(tmp_path):
    db = tmp_path / "db.sqlite"
    with_blueprints(db, ("t1", "typed target", None, None, "my notes", "u1", T0))
    service(db).handle_chat_message("t1", "m1", "M-1", XGB, T0)
    row = blueprint_row(db)
    assert row["target"] == "typed target" and row["notes"] == "my notes" and row["algorithm"] == "XGBoost"


def test_a_missing_blueprints_table_never_breaks_the_reply_or_the_saved_history(tmp_path):
    db = tmp_path / "db.sqlite"
    (reply,) = service(db).handle_chat_message("t1", "m1", "M-1", XGB, T0)      # no blueprints table here
    assert reply.startswith("Noted:")
    assert len(list_agent_messages(connector(db)(), "t1")) == 1


# ------------------------------------------------------------------ timer
def test_the_ticker_calls_its_function_repeatedly_and_stops():
    calls = []
    ticker = hook.Ticker(lambda: calls.append(1), 0.02)
    ticker.start()
    time.sleep(0.2)
    ticker.stop()
    seen = len(calls)
    time.sleep(0.1)
    assert seen >= 3 and len(calls) <= seen + 1


def test_a_failing_timer_step_does_not_kill_the_timer():
    calls = []

    def flaky():
        calls.append(1)
        raise RuntimeError("boom")

    ticker = hook.Ticker(flaky, 0.02)
    ticker.start()
    time.sleep(0.15)
    ticker.stop()
    assert len(calls) >= 2


def test_start_timer_ticks_every_team_through_the_shared_service():
    class Stub:
        def __init__(self):
            self.n = 0

        def tick_all(self):
            self.n += 1

    stub = Stub()
    hook._service, hook._ticker = stub, None
    try:
        hook.start_timer(0.02)
        hook.start_timer(0.02)                  # a second call must not start a second timer
        time.sleep(0.2)
    finally:
        hook.stop_timer()
        hook._ticker, hook._service = None, None
    assert stub.n >= 3