"""Tests for component1/agent/chat_agent.py (one team's assistant).

A pretend language model and a small model trained on simulated data are used, so no
internet and no saved model file are needed. All values and times are invented.
"""
import json

from sklearn.ensemble import RandomForestClassifier

from component1.agent.chat_agent import ChatAgent
from component1.common.schema import FieldName, FieldStatus
from component1.state.features import FEATURE_ORDER
from component1.state.simulate import FEATURE_NAMES, generate_dataset


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
    MEETING: [],
}


def scripted(prompt):
    for text, items in SCRIPT.items():
        if text in prompt:
            return json.dumps({"items": items})
    return json.dumps({"items": []})


def new_agent():
    return ChatAgent("p1", BUNDLE, scripted)


def test_a_saved_fact_is_announced_in_one_reply():
    (reply,) = new_agent().on_message("M1", "Member_A", TOURIST, now=0)
    assert reply.startswith("Noted:")
    assert "target variable = tourist arrivals" in reply
    assert "Member_A" in reply and "marked uncertain" in reply


def test_a_message_with_nothing_to_extract_gets_no_reply():
    assert new_agent().on_message("M1", "Member_C", MEETING, now=0) == []


def test_a_conflict_is_raised_when_a_second_person_disagrees():
    agent = new_agent()
    agent.on_message("M1", "Member_A", TOURIST, now=0)
    (reply,) = agent.on_message("M2", "Member_C", HOTEL, now=20)
    first, second = reply.split("\n")
    assert first.startswith("Noted:") and "conflicts with an earlier value" in first
    assert "different answers" in second
    assert "tourist arrivals" in second and "hotel occupancy" in second


def test_the_same_conflict_is_not_raised_again_on_the_next_message():
    agent = new_agent()
    agent.on_message("M1", "Member_A", TOURIST, now=0)
    agent.on_message("M2", "Member_C", HOTEL, now=20)
    assert agent.on_message("M3", "Member_B", MEETING, now=40) == []


def test_only_one_reply_is_returned_even_when_there_is_a_save_and_a_conflict():
    agent = new_agent()
    agent.on_message("M1", "Member_A", TOURIST, now=0)
    assert len(agent.on_message("M2", "Member_C", HOTEL, now=20)) == 1


def test_a_missing_field_is_not_asked_about_while_people_are_chatting():
    agent = new_agent()
    agent.on_message("M1", "Member_B", XGB, now=0)
    assert agent.tick(now=10) == []


def test_a_missing_field_is_asked_about_once_the_chat_has_gone_quiet():
    agent = new_agent()
    agent.on_message("M1", "Member_B", XGB, now=0)
    agent.on_message("M2", "Member_C", MEETING, now=20)
    (reply,) = agent.tick(now=20 + 200)
    assert "problem" in reply and "no decision yet" in reply


def test_each_missing_field_is_asked_about_once_and_one_at_a_time():
    agent = new_agent()
    agent.on_message("M1", "Member_B", XGB, now=0)
    (first,) = agent.tick(now=200)
    (second,) = agent.tick(now=400)
    assert "problem" in first
    assert "resources" in second          # the next missing field that is ready to be asked about
    assert agent.tick(now=600) == []      # nothing left that is ready, and nothing is repeated


def test_the_assistant_never_speaks_in_a_chat_nobody_has_written_in():
    assert new_agent().tick(now=10_000) == []


def test_messages_really_update_the_blueprint():
    agent = new_agent()
    agent.on_message("M1", "Member_A", TOURIST, now=0)
    agent.on_message("M2", "Member_C", HOTEL, now=20)
    f = agent.store.state.fields[FieldName.TARGET_VARIABLE]
    assert f.status == FieldStatus.CONFLICTING and f.value == "hotel occupancy"


def test_two_agents_do_not_share_anything():
    a, b = new_agent(), new_agent()
    a.on_message("M1", "Member_A", TOURIST, now=0)
    assert b.store.state.version == 0