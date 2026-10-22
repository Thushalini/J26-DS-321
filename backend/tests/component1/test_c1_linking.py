"""Tests for component1/linking: which proposal a reply answers. Simulated chats; all text invented.

A small linker is trained here, so the tests do not depend on the saved model.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from component1.agent.chat_agent import ChatAgent
from component1.common.schema import DecisionStatus
from component1.decisions.tracker import find_active
from component1.extraction.hybrid import make_hybrid_llm
from component1.linking.features import FEATURE_NAMES, Message, pair_features
from component1.linking.predict import WINDOW, candidates, link_reply
from component1.linking.simulate import generate_chats
from component1.linking.train import baseline, build, top1, train_and_evaluate
from component1.state.features import FEATURE_ORDER
from component1.state.simulate import FEATURE_NAMES as STATE_FEATURES, generate_dataset


def small_linker():
    X, y, _ = build(generate_chats(150, seed=9))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    return {"model": model, "features": FEATURE_NAMES}


def small_state_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [STATE_FEATURES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


LINKER, STATE = small_linker(), small_state_bundle()
RULES = make_hybrid_llm(None, mode="rules")


def proposal(mid, who, t, text, field, value):
    return Message(mid, who, t, text, field, value)


METRIC = proposal("p1", "M-1", 0, "How about RMSE for the metric?", "evaluation_metric", "RMSE")
ALGO = proposal("p2", "M-2", 60, "Let's use XGBoost.", "algorithm", "XGBoost")


# ---------------------------------------------------------------- features
def test_pair_features_describe_recency_speaker_and_mentions():
    reply = Message("r", "M-3", 120, "For the metric, RMSE is fine.")
    f = dict(zip(FEATURE_NAMES, pair_features(reply, METRIC, [METRIC, ALGO], 1, False)))
    assert f["is_latest_proposal"] == 0 and f["proposal_rank"] == 1
    assert f["names_the_value"] == 1 and f["names_the_field"] == 1 and f["same_speaker"] == 0
    g = dict(zip(FEATURE_NAMES, pair_features(reply, ALGO, [METRIC, ALGO], 0, False)))
    assert g["is_latest_proposal"] == 1 and g["names_the_value"] == 0 and g["names_another_value"] == 1


def test_only_recent_proposals_are_candidates():
    old = [proposal(f"p{i}", "M-1", i, "Use XGBoost", "algorithm", "XGBoost") for i in range(WINDOW + 5)]
    reply = Message("r", "M-2", 100, "agreed")
    assert len(candidates(reply, old)) == WINDOW
    late = Message("r", "M-2", 3 * 3600, "agreed")
    assert candidates(late, [METRIC]) == []                       # more than two hours later


# --------------------------------------------------------------- simulator
def test_simulated_chats_know_every_reply_s_parent_and_some_answer_older_proposals():
    chats = generate_chats(80, seed=1)
    ids = {m.message_id for c in chats for m in c.messages}
    assert all(p in {m.message_id for m in c.messages} for c in chats for p in c.parent.values())
    _, _, groups = build(chats)
    older = np.mean([not latest for _, _, latest in groups])
    assert 0.15 < older < 0.4 and len(ids) > 0
    assert [m.text for m in generate_chats(3, seed=2)[0].messages] == [m.text for m in generate_chats(3, seed=2)[0].messages]


# ---------------------------------------------------------------- training
def test_the_linker_beats_the_latest_proposal_baseline():
    r = train_and_evaluate(n_chats=200, save=False)
    best = max(r["logistic_regression"]["top1_accuracy"], r["random_forest"]["top1_accuracy"])
    assert best > r["baseline_latest_proposal"]["top1_accuracy"] + 0.05
    assert r["baseline_latest_proposal"]["top1_on_replies_to_older_proposals"] == 0.0
    assert r["data"]["kind"].startswith("SIMULATED")


def test_top1_and_baseline_scores():
    groups = [([0, 1], 1, True), ([2, 3], 0, False)]
    assert baseline(groups)["top1_accuracy"] == 0.5
    assert top1(np.array([0.1, 0.9, 0.8, 0.2]), groups)["top1_accuracy"] == 1.0


# -------------------------------------------------------------- predicting
def test_without_a_model_the_newest_proposal_is_chosen():
    assert link_reply(None, Message("r", "M-3", 120, "agreed"), [METRIC, ALGO]) is ALGO


def test_a_reply_naming_an_older_proposal_is_linked_to_it():
    reply = Message("r", "M-3", 120, "For the metric, RMSE is fine.")
    assert link_reply(LINKER, reply, [METRIC, ALGO]) is METRIC


def test_no_proposals_means_no_link():
    assert link_reply(LINKER, Message("r", "M-3", 120, "agreed"), [Message("x", "M-1", 0, "hello")]) is None


# ---------------------------------------------------------- through the chat
def agent():
    return ChatAgent("t", STATE, RULES, link_bundle=LINKER)


def test_the_reply_button_links_an_agreement_to_the_older_proposal():
    a = agent()
    a.on_message("p1", "M-1", "We will report RMSE.", 0)
    a.on_message("p2", "M-2", "Let's use XGBoost.", 60)
    a.on_message("r1", "M-3", "Agreed.", 120, reply_to="p1")
    assert find_active(a.store.state, "evaluation_metric", "RMSE").status == DecisionStatus.CONFIRMED
    assert find_active(a.store.state, "algorithm", "XGBoost").status == DecisionStatus.PROPOSED


def test_the_linker_finds_the_older_proposal_when_the_reply_names_its_field():
    a = agent()
    a.on_message("p1", "M-1", "We will report RMSE.", 0)
    a.on_message("p2", "M-2", "Let's use XGBoost.", 60)
    a.on_message("r1", "M-3", "For the metric, agreed.", 120)
    assert find_active(a.store.state, "evaluation_metric", "RMSE").status == DecisionStatus.CONFIRMED
    assert find_active(a.store.state, "algorithm", "XGBoost").status == DecisionStatus.PROPOSED


def test_a_plain_agreement_goes_to_the_newest_proposal():
    a = agent()
    a.on_message("p1", "M-1", "We will report RMSE.", 0)
    a.on_message("p2", "M-2", "Let's use XGBoost.", 60)
    a.on_message("r1", "M-3", "Agreed.", 70)
    assert find_active(a.store.state, "algorithm", "XGBoost").status == DecisionStatus.CONFIRMED


def test_a_plural_value_followed_by_a_comma_is_still_named():
    older = proposal("p1", "M-1", 0, "The dataset is SLTDA monthly arrivals.", "dataset", "SLTDA monthly arrivals")
    newer = proposal("p2", "M-2", 20, "The dataset is hotel booking records.", "dataset", "hotel booking records")
    reply = Message("r", "M-3", 40, "SLTDA monthly arrivals, agreed.")
    f_older = pair_features(reply, older, [older, newer], 1, False)
    assert f_older[FEATURE_NAMES.index("names_the_value")] == 1.0
