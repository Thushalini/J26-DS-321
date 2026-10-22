"""Tests for component1/active: field features, the simulator and the conflict-risk model.

A small model is trained here on a few simulated teams, so the tests do not depend on the saved one.
All values are invented.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from component1.active.features import FEATURE_NAMES, field_risk_features, in_dispute, staleness
from component1.active.risk import field_risks, reasons_for, risk_of
from component1.active.simulate import generate_dataset
from component1.active.train_risk import rule_score
from component1.common.events import EventStore
from component1.common.schema import FieldName, FieldStatus, Provenance, SourceType
from component1.decisions.tracker import endorse, propose


def mention(store, field, value, who, hedged=False):
    prov = Provenance(source_type=SourceType.CHAT, source_id="m", speaker=who, text_span=value)
    store.update_field(FieldName(field), value, FieldStatus.UNCERTAIN if hedged else FieldStatus.KNOWN, 0.6,
                       prov, "extractor")


@pytest.fixture(scope="module")
def data():
    return generate_dataset(15, seed=3)


@pytest.fixture(scope="module")
def bundle(data):
    X, y, _ = data
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    return {"model": model, "features": FEATURE_NAMES, "safe_reference": np.median(X[y == 0], axis=0).tolist()}


# --------------------------------------------------------------- features
def test_a_field_without_a_value_has_no_features():
    assert field_risk_features(EventStore("t").state, [], "algorithm") is None


def test_features_follow_the_decision_record():
    s = EventStore("t")
    mention(s, "algorithm", "XGBoost", "M-1", hedged=True)
    d = propose(s, "algorithm", "XGBoost", "M-1", "m1", "it fits the data")
    x = dict(zip(FEATURE_NAMES, field_risk_features(s.state, s.events, "algorithm", days_left=10)))
    assert x["latest_hedged"] == 1 and x["is_settled"] == 0 and x["has_reason"] == 1 and x["deadline_known"] == 1
    mention(s, "algorithm", "XGBoost", "M-2")
    endorse(s, s.state.decisions[d.decision_id], "M-2", finalize=True)
    x = dict(zip(FEATURE_NAMES, field_risk_features(s.state, s.events, "algorithm")))
    assert x["is_settled"] == 1 and x["is_finalized"] == 1 and x["n_agreed"] == 1 and x["deadline_known"] == 0
    assert len(x) == len(FEATURE_NAMES)


def test_staleness_counts_mentions_of_other_fields():
    s = EventStore("t")
    mention(s, "algorithm", "XGBoost", "M-1")
    for f, v in [("evaluation_metric", "RMSE"), ("dataset", "hotel data"), ("domain", "tourism")]:
        mention(s, f, v, "M-2")
    assert staleness(s.events, "algorithm") == 3 and staleness(s.events, "domain") == 0


def test_two_different_active_values_are_a_dispute():
    s = EventStore("t")
    propose(s, "algorithm", "XGBoost", "M-1", "m1")
    assert not in_dispute(s.state, "algorithm")
    propose(s, "algorithm", "LSTM", "M-2", "m2")
    assert in_dispute(s.state, "algorithm")


# -------------------------------------------------------------- simulator
def test_the_simulator_gives_rows_and_some_positives(data):
    X, y, groups = data
    assert X.shape[1] == len(FEATURE_NAMES) and len(X) == len(y) == len(groups)
    assert 0.02 < y.mean() < 0.4 and len(set(groups)) == 15


def test_the_simulator_is_repeatable():
    a, b = generate_dataset(2, seed=7), generate_dataset(2, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_the_rule_baseline_marks_unagreed_or_unsure_values():
    x = np.zeros((2, len(FEATURE_NAMES)))
    x[1, FEATURE_NAMES.index("is_settled")] = 1
    assert rule_score(x)[0] > rule_score(x)[1]


# ------------------------------------------------------------------ model
def test_an_unagreed_unsure_value_is_riskier_than_a_finalized_one(bundle):
    s = EventStore("t")
    mention(s, "target_variable", "tourist arrivals", "M-1", hedged=True)
    propose(s, "target_variable", "tourist arrivals", "M-1", "m1")
    mention(s, "algorithm", "XGBoost", "M-1")
    d = propose(s, "algorithm", "XGBoost", "M-1", "m2", "it fits the data")
    mention(s, "algorithm", "XGBoost", "M-2")
    endorse(s, s.state.decisions[d.decision_id], "M-2", finalize=True)
    risks = field_risks(bundle, s.state, s.events, days_left=20)
    assert set(risks) == {"target_variable", "algorithm"}
    assert risks["target_variable"]["risk"] > risks["algorithm"]["risk"]
    assert "nobody else has agreed with it yet" in risks["target_variable"]["reasons"]


def test_disputed_fields_get_no_score(bundle):
    s = EventStore("t")
    mention(s, "algorithm", "XGBoost", "M-1")
    propose(s, "algorithm", "XGBoost", "M-1", "m1")
    mention(s, "algorithm", "LSTM", "M-2")
    propose(s, "algorithm", "LSTM", "M-2", "m2")
    assert field_risks(bundle, s.state, s.events) == {}


def test_no_model_means_no_scores():
    assert field_risks(None, EventStore("t").state, []) == {}


def test_reasons_only_name_facts_that_raise_the_risk(bundle):
    x = list(bundle["safe_reference"])
    assert reasons_for(bundle, x) == []
    assert 0.0 <= risk_of(bundle, x) <= 1.0
