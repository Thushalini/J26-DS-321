"""Tests for the intent wiring: a message like "ok, lock that in" counts as one more person
agreeing with the value someone else just proposed. The intent model is replaced by a
pretend one, so these tests check the wiring, not how good the real model is. Values are invented."""
import json

from sklearn.ensemble import RandomForestClassifier

from component1.agent.chat_agent import ChatAgent
from component1.state.features import FEATURE_ORDER, field_features
from component1.state.simulate import FEATURE_NAMES, generate_dataset

XGB = "Let's use XGBoost, it handles tabular data well."
LOCK = "ok, lock that in"


def small_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


BUNDLE = small_bundle()


def scripted(prompt):
    if XGB in prompt:
        return json.dumps({"items": [{"field": "algorithm", "value": "XGBoost", "span": "use XGBoost", "hedged": False}]})
    return json.dumps({"items": []})


def pretend_intent(text):
    return ("finalization", 0.9) if "lock" in text else ("other", 0.9)


def agent(with_intent=True):
    a = ChatAgent("t1", BUNDLE, scripted)
    if with_intent:
        a._intent = pretend_intent
    return a


def algorithm_numbers(a):
    return field_features(a.store.events, "algorithm")      # source_count, agreement, latest_hedged, distinct_speakers


def test_a_second_person_backing_the_value_counts_as_one_more_mention():
    a = agent()
    a.on_message("m1", "M-1", XGB, 0)
    a.on_message("m2", "M-2", LOCK, 10)
    numbers = algorithm_numbers(a)
    assert numbers[0] == 2.0 and numbers[3] == 2.0           # two mentions, two different people
    assert a.store.state.fields["algorithm"].value == "XGBoost"


def test_agreeing_with_your_own_proposal_does_not_count():
    a = agent()
    a.on_message("m1", "M-1", XGB, 0)
    a.on_message("m2", "M-1", LOCK, 10)
    assert algorithm_numbers(a)[0] == 1.0


def test_without_an_intent_model_nothing_changes():
    a = agent(with_intent=False)
    a.on_message("m1", "M-1", XGB, 0)
    a.on_message("m2", "M-2", LOCK, 10)
    assert algorithm_numbers(a)[0] == 1.0


def test_an_old_proposal_is_no_longer_the_topic():
    # Reply linking looks back over the last 15 messages (linking/predict.py WINDOW); older is too far.
    a = agent()
    a.on_message("m1", "M-1", XGB, 0)
    for i in range(16):
        a.on_message(f"f{i}", "M-2", "what time is the lab tomorrow", 10 + i)
    a.on_message("m9", "M-2", LOCK, 100)
    assert algorithm_numbers(a)[0] == 1.0


def test_a_reply_after_some_small_talk_still_reaches_the_only_proposal():
    a = agent()
    a.on_message("m1", "M-1", XGB, 0)
    for i in range(6):
        a.on_message(f"f{i}", "M-2", "what time is the lab tomorrow", 10 + i)
    a.on_message("m9", "M-2", LOCK, 100)
    assert algorithm_numbers(a)[0] == 2.0

def restarted(old):
    """A new agent rebuilt from the saved history of an old one, like after a server restart."""
    store = type(old.store).from_events("t1", old.store.events)
    new = ChatAgent("t1", BUNDLE, scripted, store=store)
    new._intent = pretend_intent
    return new


def test_after_a_restart_the_last_mention_is_the_topic_again():
    a = agent()
    a.on_message("m1", "M-1", XGB, 0)
    b = restarted(a)
    b.on_message("m2", "M-2", LOCK, 10)
    assert algorithm_numbers(b)[0] == 2.0


def test_after_a_restart_the_topic_counts_for_one_message_only():
    a = agent()
    a.on_message("m1", "M-1", XGB, 0)
    b = restarted(a)
    b.on_message("m2", "M-2", "what time is the lab tomorrow", 10)
    b.on_message("m3", "M-2", LOCK, 20)
    assert algorithm_numbers(b)[0] == 1.0


def test_after_a_restart_you_still_cannot_back_your_own_proposal():
    a = agent()
    a.on_message("m1", "M-1", XGB, 0)
    b = restarted(a)
    b.on_message("m2", "M-1", LOCK, 10)
    assert algorithm_numbers(b)[0] == 1.0