"""Tests for component1/state (feature 4). All data here is synthetic or hand-built."""
import numpy as np

from component1.state.simulate import (
    FEATURE_NAMES,
    LABELS,
    Mention,
    generate_dataset,
    label_from_mentions,
)
from component1.state.train import rule_baseline, train_and_evaluate


def m(kind, speaker, value="A", hedged=False, confidence=0.9):
    return Mention(kind, speaker, value, hedged, confidence)


# ------------------------------------------- the codebook rules (Task 2)
def test_one_plain_statement_is_known():
    assert label_from_mentions([m("state", 0)]) == "known"


def test_one_hedged_statement_is_uncertain():
    assert label_from_mentions([m("state", 0, hedged=True)]) == "uncertain"


def test_agreement_from_another_person_makes_it_confirmed():
    assert label_from_mentions([m("state", 0), m("endorse", 1)]) == "confirmed"


def test_a_hedged_statement_that_someone_agrees_with_is_confirmed():
    assert label_from_mentions([m("state", 0, hedged=True), m("endorse", 1)]) == "confirmed"


def test_saying_it_again_yourself_is_not_confirmation():
    assert label_from_mentions([m("state", 0), m("repeat", 0)]) == "known"


def test_agreeing_with_yourself_is_not_confirmation():
    assert label_from_mentions([m("state", 0), m("endorse", 0)]) == "known"


def test_two_different_values_from_different_people_is_conflicting():
    assert label_from_mentions([m("state", 0, "A"), m("state", 1, "B")]) == "conflicting"


def test_a_conflict_that_someone_then_settles_is_confirmed():
    assert label_from_mentions([m("state", 0, "A"), m("state", 1, "B"), m("endorse", 2, "B")]) == "confirmed"


def test_a_contradiction_after_an_agreement_reopens_the_conflict():
    assert label_from_mentions([m("state", 0, "A"), m("endorse", 1, "A"), m("state", 2, "B")]) == "conflicting"


# ------------------------------------------------------- simulated data
def test_the_dataset_has_the_right_shape_labels_and_ranges():
    X, y = generate_dataset(400, seed=1)
    assert X.shape == (400, len(FEATURE_NAMES))
    assert set(y) <= set(LABELS)
    assert X[:, 0].min() >= 1                                   # source_count
    assert 0.5 <= X[:, 1].min() and X[:, 1].max() <= 1.0        # agreement
    assert 0.0 <= X[:, 2].min() and X[:, 2].max() <= 1.0        # mean_confidence
    assert set(np.unique(X[:, 4])) <= {0.0, 1.0}                # latest_hedged
    assert set(np.unique(X[:, 5])) <= {1.0, 2.0, 3.0}           # distinct_speakers


def test_the_same_seed_gives_the_same_data_and_all_classes_appear():
    X1, y1 = generate_dataset(2000, seed=7)
    X2, y2 = generate_dataset(2000, seed=7)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert set(y1) == set(LABELS)


# --------------------------------------------------------------- baseline
def test_the_fixed_rule_baseline_gives_sensible_labels():
    rows = np.array([
        [1, 1.0, 0.90, 10, 0, 1],   # one plain statement
        [2, 1.0, 0.90, 10, 0, 2],   # two agreeing sources
        [2, 0.5, 0.90, 10, 0, 2],   # sources disagree
        [1, 1.0, 0.60, 10, 0, 1],   # low confidence
    ])
    assert list(rule_baseline(rows)) == ["known", "confirmed", "conflicting", "uncertain"]


# ----------------------------------------------------------------- training
def test_training_runs_and_the_extra_features_help():
    r = train_and_evaluate(n=1500, n_estimators=30, save=False)
    assert "SYNTHETIC" in r["data"]["kind"]
    four = r["feature_sets"]["proposal_4_features"]["random_forest"]["macro_f1"]
    six = r["feature_sets"]["extended_6_features"]["random_forest"]["macro_f1"]
    assert six > four + 0.1
    assert six > 0.6


def test_the_model_ignores_the_pure_noise_feature():
    r = train_and_evaluate(n=1500, n_estimators=30, save=False)
    importance = r["feature_sets"]["extended_6_features"]["random_forest_feature_importance"]
    assert importance["recency_minutes"] < 0.15