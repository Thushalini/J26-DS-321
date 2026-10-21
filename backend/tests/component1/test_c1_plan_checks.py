"""Tests for component1/checks: do the parts of the plan fit, and what is suggested instead.

Blueprints are built with the real pipeline (rules only) and a small model trained on simulated
data, or set field by field. All values are invented; fixed dates keep the tests independent of today.
"""
from datetime import date

import pytest
from sklearn.ensemble import RandomForestClassifier

from component1.agent.policy import Buddy
from component1.agent.signals import CHECK
from component1.checks.plan_checks import check_signals, feasibility_of, record_checks, run_checks
from component1.common.events import EventStore
from component1.common.schema import FieldName, FieldStatus, Provenance, SourceType
from component1.extraction.hybrid import make_hybrid_llm
from component1.extraction.rules import canonical_names
from component1.pipeline import process_message, state_signals
from component1.plan import plan_from_events
from component1.state.features import FEATURE_ORDER
from component1.state.simulate import FEATURE_NAMES, generate_dataset

RULES = make_hybrid_llm(None, mode="rules")


def small_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


BUNDLE = small_bundle()


def plan(status=FieldStatus.KNOWN, **values):
    """A store with the given fields set, e.g. plan(task_type="classification", evaluation_metric="RMSE")."""
    store = EventStore("t")
    for field, value in values.items():
        prov = Provenance(source_type=SourceType.CHAT, source_id="m", speaker="M-1", text_span=value)
        store.update_field(FieldName(field), value, status, 0.5, prov, "extractor")
    return store


def results(store, today=None):
    return {r.key: r for r in run_checks(store.state, store.events, today)}


# ---------------------------------------------------------------- names
@pytest.mark.parametrize("kind, text, names", [
    ("algorithm", "an xgboost classifier", ["XGBoost"]),
    ("task", "logistic regression for classification", ["classification"]),   # not the task "regression"
    ("metric", "f1 score", ["F1-score"]),
    ("resource", "we'll train on google colab", ["Google Colab"]),
    ("algorithm", "something custom", []),
])
def test_values_are_recognised_in_standard_spelling(kind, text, names):
    assert canonical_names(kind, text) == names


# ------------------------------------------------------------ metric/task
def test_a_regression_metric_for_a_classification_task_fails_with_alternatives():
    r = results(plan(task_type="classification", evaluation_metric="RMSE"))["metric_fits_task"]
    assert r.status == "fail" and r.field == "evaluation_metric"
    assert r.alternatives == ("F1-score", "ROC AUC")
    assert r.detail == ("RMSE is a regression metric, but the task is classification. "
                        "Consider F1-score or ROC AUC instead.")


@pytest.mark.parametrize("task, metric", [("regression", "RMSE"), ("classification", "F1-score"),
                                          ("forecasting", "MAPE"), ("clustering", "Silhouette Score")])
def test_matching_metrics_pass(task, metric):
    assert results(plan(task_type=task, evaluation_metric=metric))["metric_fits_task"].status == "pass"


# --------------------------------------------------------- algorithm/task
@pytest.mark.parametrize("task, algorithm, words", [
    ("classification", "K-Means", "K-Means is a clustering method"),
    ("classification", "ARIMA", "ARIMA is a time-series forecasting method"),
    ("regression", "Logistic Regression", "Logistic Regression is a classification method"),
    ("clustering", "XGBoost", "XGBoost is not usually used for clustering"),
])
def test_algorithms_that_do_not_fit_the_task_fail(task, algorithm, words):
    r = results(plan(task_type=task, algorithm=algorithm))["algorithm_fits_task"]
    assert r.status == "fail" and r.detail.startswith(words) and r.alternatives


@pytest.mark.parametrize("task, algorithm", [("classification", "XGBoost"), ("regression", "Random Forest"),
                                             ("forecasting", "ARIMA"), ("forecasting", "LSTM"),
                                             ("clustering", "K-Means")])
def test_algorithms_that_fit_pass(task, algorithm):
    assert results(plan(task_type=task, algorithm=algorithm))["algorithm_fits_task"].status == "pass"


# ------------------------------------------------------------ target/task
def test_a_number_like_target_with_classification_is_a_warning():
    r = results(plan(task_type="classification", target_variable="tourist arrivals"))["target_fits_task"]
    assert r.status == "warning" and r.alternatives == ("regression", "forecasting")


def test_a_category_like_target_with_regression_is_a_warning():
    r = results(plan(task_type="regression", target_variable="customer churn"))["target_fits_task"]
    assert r.status == "warning" and r.alternatives == ("classification",)


def test_targets_that_fit_pass_and_unclear_targets_are_not_judged():
    assert results(plan(task_type="classification", target_variable="churn"))["target_fits_task"].status == "pass"
    assert "target_fits_task" not in results(plan(task_type="classification", target_variable="outcome"))


# -------------------------------------------------------- compute / time
def test_deep_learning_without_compute_is_a_warning_and_a_gpu_makes_it_pass():
    r = results(plan(task_type="classification", algorithm="LSTM"))["compute_fits_model"]
    assert r.status == "warning" and r.field == "resources" and r.alternatives[0] == "Google Colab"
    assert results(plan(task_type="classification", algorithm="LSTM", resources="GPU"))["compute_fits_model"].status == "pass"
    assert "compute_fits_model" not in results(plan(task_type="classification", algorithm="XGBoost"))


def test_deep_learning_with_little_time_left_is_a_warning():
    store = plan(task_type="classification", algorithm="BERT", resources="GPU", timeline="deadline 2026-11-01")
    r = results(store, today=date(2026, 10, 19))["timeline_fits_model"]
    assert r.status == "warning" and "only 13 days are left" in r.detail
    assert results(store, today=date(2026, 9, 1))["timeline_fits_model"].status == "pass"
    assert "the deadline has passed" in results(store, today=date(2026, 11, 5))["timeline_fits_model"].detail


# -------------------------------------------------------------- what runs
def test_nothing_is_checked_until_the_fields_it_needs_have_a_value():
    assert run_checks(EventStore("t").state) == []
    assert run_checks(plan(evaluation_metric="RMSE").state) == []          # no task yet


def test_a_field_the_team_disagrees_about_is_not_checked():
    store = plan(status=FieldStatus.CONFLICTING, task_type="classification")
    store.update_field(FieldName.EVALUATION_METRIC, "RMSE", FieldStatus.KNOWN, 0.5,
                       Provenance(source_type=SourceType.CHAT, source_id="m", speaker="M-1", text_span="RMSE"),
                       "extractor")
    assert "metric_fits_task" not in results(store)


def test_unknown_names_are_not_judged():
    assert "algorithm_fits_task" not in results(plan(task_type="classification", algorithm="our own model"))


# ------------------------------------------------------------- recording
def test_results_are_stored_as_feasibility_only_when_they_change():
    store = plan(task_type="classification", evaluation_metric="RMSE")
    record_checks(store)
    assert store.state.feasibility == {"metric_fits_task": "fail"}
    versions = store.state.version
    record_checks(store)
    assert store.state.version == versions                                  # nothing new to store
    assert feasibility_of(run_checks(store.state)) == {"metric_fits_task": "fail"}


def test_the_pipeline_stores_checks_after_a_message_and_clears_them_when_fixed():
    store = EventStore("t")
    process_message(store, BUNDLE, RULES, "m1", "M-1", "This is a classification task and we report RMSE.")
    assert store.state.feasibility == {"metric_fits_task": "fail"}
    process_message(store, BUNDLE, RULES, "m2", "M-2", "This is a classification task and we report F1 score.")
    # RMSE and F1 now disagree, so the metric is not checked until the team settles it
    assert store.state.feasibility == {}


# ----------------------------------------------------------------- Buddy
def test_buddy_names_the_problem_and_the_alternatives_and_does_not_repeat_it():
    store = plan(task_type="classification", evaluation_metric="RMSE")
    signals = [s for s in state_signals(store) if s.kind == CHECK]
    assert len(signals) == 1
    buddy = Buddy()
    (text,) = buddy.step(signals, now=0)
    assert text == ("Possible problem with the evaluation metric: RMSE is a regression metric, but the task is "
                    "classification. Consider F1-score or ROC AUC instead. Is that intended?")
    assert buddy.step(signals, now=200) == []


def test_passing_checks_tell_buddy_nothing():
    assert check_signals(run_checks(plan(task_type="regression", evaluation_metric="RMSE").state)) == []


# ------------------------------------------------------------------ plan
def test_the_plan_lists_the_checks_and_the_stored_feasibility():
    store = EventStore("t")
    process_message(store, BUNDLE, RULES, "m1", "M-1", "It is a classification task; we will use K-Means.")
    out = plan_from_events("t", store.events)
    (check,) = out["checks"]
    assert check["key"] == "algorithm_fits_task" and check["label"] == "Algorithm fits the task"
    assert check["status"] == "fail" and check["alternatives"] == ["Random Forest", "XGBoost"]
    assert out["feasibility"] == {"algorithm_fits_task": "fail"}


def test_component_4_turns_the_stored_checks_into_a_feasibility_score():
    pytest.importorskip("component4")
    from component4.m1_feature_integration.c1_adapter import project_features

    store = plan(task_type="classification", evaluation_metric="RMSE", algorithm="XGBoost")
    record_checks(store)
    features = project_features(store.state.model_dump(mode="json"))["features"]
    assert features["feasibility_score"] == 0.5                             # one pass (1.0) and one fail (0.0)
