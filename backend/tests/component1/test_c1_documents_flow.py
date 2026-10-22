"""Tests for documents end to end in the service: upload -> suggestions -> Buddy -> accept / dismiss.

SQLite, rules-only extraction and a small model trained on simulated data, so no internet, key
or saved model file is needed. All text is invented.
"""
import sqlite3

import pytest
from sklearn.ensemble import RandomForestClassifier

from component1.common.schema import FieldName, FieldStatus, SourceType
from component1.extraction.hybrid import make_hybrid_llm
from component1.service import AgentService
from component1.state.features import FEATURE_ORDER, mentions_for_field
from component1.state.simulate import FEATURE_NAMES, generate_dataset
from component1.storage import create_document, get_document, list_doc_facts, load_events


def small_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


BUNDLE = small_bundle()
T0 = 1_800_000_000.0

PLAN = (
    "Project plan\n\n"
    "Our project is in the tourism domain.\n\n"
    "We will use the Kaggle tourism dataset.\n\n"
    "We will predict tourist arrivals using ARIMA and report RMSE.\n\n"
    "References\n\n[1] A study that used LSTM on other data."
).encode("utf-8")


def connector(path):
    def connect():
        c = sqlite3.connect(str(path))
        c.execute("CREATE TABLE IF NOT EXISTS blueprints (team_id TEXT PRIMARY KEY, target TEXT, algorithm TEXT, "
                  "metric TEXT, notes TEXT, updated_by TEXT, updated_at TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS activities (activity_id TEXT, team_id TEXT)")
        return c
    return connect


@pytest.fixture
def db(tmp_path):
    return tmp_path / "app.db"


def service(db, **kwargs):
    return AgentService(bundle=BUNDLE, llm=make_hybrid_llm(None, mode="rules"), env_file=None,
                        conn_factory=connector(db), **kwargs)


def upload(svc, db, data=PLAN, name="Plan.txt", team="t1"):
    conn = connector(db)()
    from component1.storage import ensure_c1_tables
    ensure_c1_tables(conn)
    doc = create_document(conn, team, name, len(data), uploaded_by="u1")
    conn.close()
    return doc["doc_id"], svc.process_document(team, doc["doc_id"], name, data)


def open_facts(db, team="t1"):
    return [f for f in list_doc_facts(connector(db)(), team) if f["status"] == "open"]


def fact_for(db, field, team="t1"):
    return next(f for f in open_facts(db, team) if f["field"] == field)


# ------------------------------------------------------------------ upload
def test_an_upload_becomes_suggestions_and_the_reference_list_is_ignored(db):
    svc = service(db)
    doc_id, summary = upload(svc, db)
    assert summary["status"] == "done" and summary["facts"] >= 4
    doc = get_document(connector(db)(), "t1", doc_id)
    assert doc["status"] == "done" and doc["facts"] == summary["facts"]
    values = {(f["field"], f["value"]) for f in open_facts(db)}
    assert {("domain", "tourism"), ("dataset", "Kaggle tourism dataset"), ("algorithm", "ARIMA"),
            ("evaluation_metric", "RMSE")} <= values
    assert all(v != "LSTM" for _, v in values)


def test_an_unreadable_upload_is_marked_failed_with_a_reason(db):
    svc = service(db)
    doc_id, summary = upload(svc, db, data=b"%PDF-1.4 broken", name="bad.pdf")
    assert summary["status"] == "failed"
    doc = get_document(connector(db)(), "t1", doc_id)
    assert doc["status"] == "failed" and "could not be read" in doc["error"]


def test_suggestions_do_not_change_the_blueprint_by_themselves(db):
    svc = service(db)
    upload(svc, db)
    assert load_events(connector(db)(), "t1") == []


# ------------------------------------------------------------------ Buddy
def test_buddy_offers_a_suggestion_after_an_upload_even_before_any_chat(db):
    svc = service(db)
    upload(svc, db)
    (reply,) = svc.tick("t1", T0)
    assert reply.startswith("I found a possible dataset in Plan.txt: Kaggle tourism dataset")


def test_a_suggestion_is_not_offered_for_a_field_the_chat_already_settled(db):
    svc = service(db)
    svc.handle_chat_message("t1", "m1", "M-1", "We will use the UCI Online Retail dataset.", T0)
    upload(svc, db)
    replies = svc.tick("t1", T0 + 200)
    assert replies and "possible dataset" not in replies[0]


# ----------------------------------------------------------------- accept
def test_accepting_records_the_document_and_the_member_as_two_sources(db):
    svc = service(db)
    upload(svc, db)
    f = fact_for(db, "algorithm")
    result = svc.accept_suggestion("t1", f["fact_id"], "M-2")
    assert result["ok"] and result["value"] == "ARIMA" and result["status"] in ("known", "confirmed")

    events = load_events(connector(db)(), "t1")
    mentions = mentions_for_field(events, "algorithm")
    assert [m["speaker"] for m in mentions] == ["doc:Plan.txt", "M-2"]
    doc_prov = next(e.payload["provenance"] for e in events if e.payload.get("provenance"))
    assert doc_prov["source_type"] == SourceType.DOCUMENT.value and doc_prov["text_span"] in PLAN.decode()

    assert all(x["status"] == "accepted" for x in list_doc_facts(connector(db)(), "t1") if x["field"] == "algorithm")
    row = connector(db)().execute("SELECT algorithm FROM blueprints WHERE team_id = 't1'").fetchone()
    assert row == ("ARIMA",)                       # Component 2's drift row follows the plan


def test_a_suggestion_cannot_be_accepted_twice(db):
    svc = service(db)
    upload(svc, db)
    f = fact_for(db, "algorithm")
    assert svc.accept_suggestion("t1", f["fact_id"], "M-2")["ok"]
    again = svc.accept_suggestion("t1", f["fact_id"], "M-1")
    assert not again["ok"] and "already" in again["error"]


def test_an_accepted_value_that_disagrees_with_the_chat_shows_as_a_conflict(db):
    svc = service(db)
    svc.handle_chat_message("t1", "m1", "M-1", "Let's use XGBoost for this.", T0)
    upload(svc, db)
    svc.accept_suggestion("t1", fact_for(db, "algorithm")["fact_id"], "M-2")
    agent = svc._agents["t1"]
    assert agent.store.state.fields[FieldName.ALGORITHM].status == FieldStatus.CONFLICTING


# ---------------------------------------------------------------- dismiss
def test_a_dismissed_suggestion_is_never_offered_again(db):
    svc = service(db)
    upload(svc, db)
    f = fact_for(db, "dataset")
    assert svc.dismiss_suggestion("t1", f["fact_id"], "M-1")["ok"]
    assert "dataset" not in svc._agents["t1"].candidates
    upload(svc, db, name="Plan v2.txt")                       # the same value again
    assert "dataset" not in svc._agents["t1"].candidates
    (reply,) = svc.tick("t1", T0)
    assert "possible dataset" not in reply


# ------------------------------------------------------------ chat "yes"
def test_a_yes_from_a_member_after_buddy_asks_accepts_the_suggestion(db, monkeypatch):
    monkeypatch.setattr("component1.intent.predict.predict_intent",
                        lambda bundle, text, prev="": ("agreement", 0.9) if "yes" in text.lower() else ("other", 0.9))
    svc = service(db)
    svc._intent_tried, svc._intent_bundle = True, {"stand-in": True}
    upload(svc, db)
    (question,) = svc.tick("t1", T0)
    assert "possible dataset" in question

    (reply,) = svc.handle_chat_message("t1", "m1", "M-2", "Yes, use that.", T0 + 5)
    assert "dataset = Kaggle tourism dataset" in reply and "accepted by M-2" in reply
    assert all(x["status"] == "accepted" for x in list_doc_facts(connector(db)(), "t1") if x["field"] == "dataset")
    speakers = [m["speaker"] for m in mentions_for_field(load_events(connector(db)(), "t1"), "dataset")]
    assert speakers == ["doc:Plan.txt", "M-2"]


def test_without_the_intent_model_a_short_yes_to_buddys_question_still_accepts_it(db):
    svc = service(db)
    upload(svc, db)
    svc.tick("t1", T0)
    svc.handle_chat_message("t1", "m1", "M-2", "Yes, use that.", T0 + 5)
    assert any(x["status"] == "accepted" for x in list_doc_facts(connector(db)(), "t1") if x["field"] == "dataset")


def test_without_the_intent_model_a_longer_reply_changes_nothing(db):
    svc = service(db)
    upload(svc, db)
    svc.tick("t1", T0)
    svc.handle_chat_message("t1", "m1", "M-2", "Hmm, I am not sure about that one.", T0 + 5)
    assert all(x["status"] == "open" for x in list_doc_facts(connector(db)(), "t1") if x["field"] == "dataset")


# ---------------------------------------------------------------- restart
def test_after_a_restart_open_suggestions_come_back_and_decided_ones_do_not(db):
    first = service(db)
    upload(first, db)
    first.accept_suggestion("t1", fact_for(db, "algorithm")["fact_id"], "M-2")
    first.dismiss_suggestion("t1", fact_for(db, "dataset")["fact_id"], "M-1")

    second = service(db)                          # a fresh service reading the same database
    reply = second.tick("t1", T0)                 # no agent yet: the timer only ticks known teams
    assert reply == []
    second.handle_chat_message("t1", "m9", "M-1", "hello team", T0)
    agent = second._agents["t1"]
    assert "algorithm" not in agent.candidates and "dataset" not in agent.candidates
    assert "evaluation_metric" in agent.candidates
    assert agent.store.state.fields[FieldName.ALGORITHM].value == "ARIMA"


def test_documents_without_a_database_fail_politely():
    svc = AgentService(bundle=BUNDLE, llm=make_hybrid_llm(None, mode="rules"), env_file=None)
    assert svc.process_document("t1", "d1", "Plan.txt", PLAN)["status"] == "failed"
    assert not svc.accept_suggestion("t1", "f1", "M-1")["ok"]
    assert not svc.dismiss_suggestion("t1", "f1", "M-1")["ok"]
