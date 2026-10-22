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


# ------------------------------------------------------------ specificity
from component1.active.specific import clarify_question, vague_fields  # noqa: E402


@pytest.mark.parametrize("field, value", [
    ("algorithm", "Neural Network"), ("algorithm", "machine learning"), ("evaluation_metric", "performance"),
    ("task_type", "prediction"), ("dataset", "Kaggle data"), ("target_variable", "the output"),
    ("domain", "data science"), ("problem", "build a model"), ("resources", "the cloud"), ("timeline", "soon"),
])
def test_vague_values_get_a_question(field, value):
    assert clarify_question(field, value)


@pytest.mark.parametrize("field, value", [
    ("algorithm", "XGBoost"), ("evaluation_metric", "RMSE"), ("task_type", "regression"),
    ("dataset", "Kaggle hotel bookings dataset"), ("target_variable", "tourist arrivals"), ("domain", "tourism"),
    ("problem", "predicting tourist arrivals for Sri Lanka"), ("resources", "CPU only"),
    ("timeline", "deadline in 2 weeks"), ("timeline", "week 10"),
])
def test_specific_values_get_none(field, value):
    assert clarify_question(field, value) is None


def test_the_question_names_the_source_and_fits_the_task():
    assert "Which Kaggle dataset exactly?" in clarify_question("dataset", "data from Kaggle")
    assert "RMSE or MAE" in clarify_question("evaluation_metric", "error", task="regression")


def test_vague_fields_skips_missing_and_disputed_fields():
    s = EventStore("t")
    mention(s, "algorithm", "Neural Network", "M-1")
    mention(s, "dataset", "Kaggle data", "M-1")
    s.update_field(FieldName.DATASET, "Kaggle data", FieldStatus.CONFLICTING, 0.5, None, "state_classifier")
    assert set(vague_fields(s.state)) == {"algorithm"}


# --------------------------------------------------------------- priority
from component1.active.priority import impact, question_signals, urgency  # noqa: E402
from component1.active.simulate import simulate_team  # noqa: E402
from component1.agent.policy import Buddy, clarify_text, confirm_text  # noqa: E402
from component1.agent.signals import CHECK, CLARIFY, CONFIRM, CONFLICT, GAP, Signal  # noqa: E402
from component1.pipeline import state_signals  # noqa: E402


class FixedRisk:
    """A pretend risk model that gives every field the same risk."""
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]] * len(X))


def risky(p):
    return {"model": FixedRisk(p), "features": FEATURE_NAMES, "safe_reference": [0.0] * len(FEATURE_NAMES)}


def test_impact_counts_the_fields_that_depend_on_it():
    assert impact("dataset") > impact("domain") and impact("problem") > impact("resources")


def test_urgency_rises_near_the_deadline_mostly_for_build_fields():
    assert urgency("algorithm", None, 0) == 1.0
    assert urgency("algorithm", 3, 0) > urgency("domain", 3, 0) > 1.0
    assert urgency("algorithm", 60, 0) == 1.0


def test_gap_clarify_and_confirm_questions_are_scored():
    s = EventStore("t")
    mention(s, "problem", "forecasting arrivals", "M-1")
    propose(s, "problem", "forecasting arrivals", "M-1", "m1")
    mention(s, "algorithm", "Neural Network", "M-1")
    propose(s, "algorithm", "Neural Network", "M-1", "m2")
    qs = question_signals(s.state, s.events, risky(0.5))
    kinds = {(q.kind, q.field) for q in qs}
    assert (CLARIFY, "algorithm") in kinds                       # vague: asked to clarify, not to confirm
    assert (CONFIRM, "algorithm") not in kinds and (CONFIRM, "problem") in kinds
    assert (GAP, "dataset") in kinds
    assert [q.priority for q in qs] == sorted((q.priority for q in qs), reverse=True)


def test_low_risk_fields_and_warned_fields_get_no_question():
    s = EventStore("t")
    mention(s, "problem", "forecasting arrivals", "M-1")
    propose(s, "problem", "forecasting arrivals", "M-1", "m1")
    assert not any(q.kind == CONFIRM for q in question_signals(s.state, s.events, risky(0.1)))
    assert not any(q.field == "problem" for q in question_signals(s.state, s.events, risky(0.9), skip={"problem"}))


def test_buddy_asks_the_highest_priority_question_after_conflicts():
    asks = [Signal(GAP, "domain", priority=1.0), Signal(CLARIFY, "algorithm", ("Neural Network",),
                                                         detail="Which model exactly?", priority=2.0)]
    b = Buddy()
    assert "A detail is missing: Which model exactly?" in b.step(asks, 1000)[0]
    b = Buddy()
    reply = b.step(asks + [Signal(CONFLICT, "dataset", ("a", "b"))], 1000)[0]
    assert "different answers" in reply


def test_unscored_questions_keep_the_old_order():
    b = Buddy()
    assert "objective" in b.step([Signal(GAP, "objective"), Signal(GAP, "dataset")], 1000)[0]


def test_confirm_and_clarify_wording():
    c = Signal(CONFIRM, "target_variable", ("tourist arrivals",),
               detail="only one person has mentioned it; the last mention sounded unsure")
    assert confirm_text(c) == ("Can someone confirm the target variable: tourist arrivals? I ask because only one "
                               "person has mentioned it and the last mention sounded unsure.")
    assert clarify_text(Signal(CLARIFY, "dataset", detail="Which dataset exactly?")) == \
        "A detail is missing: Which dataset exactly?"


def test_the_switch_turns_the_new_questions_off(monkeypatch):
    s = EventStore("t")
    mention(s, "problem", "forecasting arrivals", "M-1")
    propose(s, "problem", "forecasting arrivals", "M-1", "m1")
    mention(s, "algorithm", "Neural Network", "M-1")
    propose(s, "algorithm", "Neural Network", "M-1", "m2")
    assert any(x.kind == CLARIFY for x in state_signals(s))
    monkeypatch.setenv("C1_ACTIVE", "off")
    assert not any(x.kind in (CLARIFY, CONFIRM) or x.priority for x in state_signals(s))


def test_a_plan_check_warning_replaces_a_second_question_about_the_same_field():
    s = EventStore("t")
    for f, v in [("task_type", "regression"), ("evaluation_metric", "performance")]:
        mention(s, f, v, "M-1")
        propose(s, f, v, "M-1", f)
    sig = state_signals(s)
    warned = {x.field for x in sig if x.kind == CHECK}
    assert not any(x.kind in (CLARIFY, CONFIRM) and x.field in warned for x in sig)


# ------------------------------------------------------- question strategy
def test_an_asked_team_reports_its_outcome():
    out = {}
    simulate_team(np.random.default_rng(4), asker=lambda st, ev, d: ("gap", "problem"), vague_rate=0.25,
                  outcome=out, snapshots=False)
    assert {"rework", "changes", "asked", "completed_at", "steps", "vague_left"} <= set(out)
    assert out["asked"] > 0


# ------------------------------------------------------------ sensitivity
from component1.active.sensitivity import _describe, feasibility_dependencies  # noqa: E402
from component1.agent.policy import gap_text  # noqa: E402


class FakeFeasibility:
    """A pretend feasibility model: feasible with 5,000+ rows, infeasible below; a GPU does not matter."""
    classes_ = ["feasible", "infeasible", "risky"]


@pytest.fixture
def feasibility(monkeypatch):
    import component1.active.sensitivity as sens

    def judge(bundle, task, algorithm, rows, has_gpu=False, weeks_left=None):
        label = "feasible" if rows >= 5000 else "infeasible"
        return {"label": label, "probabilities": {"feasible": 0.9 if label == "feasible" else 0.1}}
    monkeypatch.setattr(sens, "judge", judge)
    return FakeFeasibility()


def plan_store(**values):
    s = EventStore("t")
    for f, (v, hedged) in values.items():
        mention(s, f, v, "M-1", hedged=hedged)
        propose(s, f, v, "M-1", f)
    return s


def test_a_missing_size_that_flips_the_verdict_is_named(feasibility):
    s = plan_store(task_type=("regression", False), algorithm=("XGBoost", False))
    deps = feasibility_dependencies(feasibility, s.state, s.events)
    assert [d["field"] for d in deps] == ["dataset_size"] and deps[0]["known"] == "missing"
    assert deps[0]["text"] == ("whether the plan is feasible depends on the dataset size: it looks infeasible "
                               "with about 500 rows and feasible with about 5,000 or more")


def test_a_clearly_stated_size_is_not_second_guessed(feasibility):
    s = plan_store(task_type=("regression", False), algorithm=("XGBoost", False), dataset_size=("500 rows", False))
    assert feasibility_dependencies(feasibility, s.state, s.events) == []


def test_an_unsure_size_is_tried_too(feasibility):
    s = plan_store(task_type=("regression", False), algorithm=("XGBoost", False), dataset_size=("500 rows", True))
    assert [d["known"] for d in feasibility_dependencies(feasibility, s.state, s.events)] == ["unsure"]


def test_nothing_without_an_algorithm_or_a_model(feasibility):
    s = plan_store(task_type=("regression", False))
    assert feasibility_dependencies(feasibility, s.state, s.events) == []
    assert feasibility_dependencies(None, s.state, s.events) == []


def test_the_gap_question_says_why_it_matters(feasibility):
    s = plan_store(problem=("forecasting hotel demand", False), dataset=("Kaggle hotel bookings", False),
                   task_type=("regression", False), algorithm=("XGBoost", False))
    qs = question_signals(s.state, s.events, None, feasibility_bundle=feasibility)
    size = next(q for q in qs if q.field == "dataset_size")
    plain = next(q for q in question_signals(s.state, s.events, None) if q.field == "dataset_size")
    assert size.priority == pytest.approx(plain.priority * 1.5)
    assert gap_text(size).endswith("It matters because whether the plan is feasible depends on the dataset size: "
                                   "it looks infeasible with about 500 rows and feasible with about 5,000 or more.")
    assert "It matters" not in gap_text(plain)


def test_an_unsure_decisive_value_gets_a_confirm_question(feasibility):
    s = plan_store(task_type=("regression", False), algorithm=("XGBoost", False), dataset_size=("500 rows", True))
    qs = question_signals(s.state, s.events, risky(0.05), feasibility_bundle=feasibility)
    c = next(q for q in qs if q.kind == CONFIRM and q.field == "dataset_size")
    assert c.detail.startswith("whether the plan is feasible depends on it")


def test_values_with_the_same_verdict_are_said_once():
    s = lambda *xs: [{"value": v, "label": l} for v, l in xs]
    assert _describe("rows", s((500, "infeasible"), (1000, "infeasible"), (5000, "feasible"))) == \
        "infeasible with about 500 to 1,000 rows and feasible with about 5,000"
    assert _describe("weeks_left", s((2, "risky"), (6, "feasible"), (12, "feasible"))) == \
        "risky with 2 weeks left and feasible with 6 or more"
    assert _describe("has_gpu", s((False, "infeasible"), (True, "risky"))) == "infeasible without a GPU but risky with one"


# ------------------------------------------------------------- the screen
from component1.plan import plan_from_events  # noqa: E402


def test_the_plan_lists_what_to_settle_next_and_tags_fields(monkeypatch):
    import component1.pipeline as pipeline
    monkeypatch.setattr(pipeline, "risk_bundle", lambda: risky(0.5))
    monkeypatch.setattr(pipeline, "feasibility_bundle", lambda: None)
    s = plan_store(problem=("forecasting hotel demand", False), target_variable=("hotel demand", True),
                   algorithm=("Neural Network", False))
    p = plan_from_events("t", list(s.events))
    nq = p["next_questions"]
    assert 0 < len(nq) <= 3 and [q["priority"] for q in nq] == sorted((q["priority"] for q in nq), reverse=True)
    assert {"kind", "action", "field", "label", "value", "why", "priority"} <= set(nq[0])
    fields = {f["name"]: f for f in p["fields"]}
    assert fields["algorithm"]["needs_detail"].startswith("The algorithm is 'Neural Network'")
    assert "at_risk" in fields["target_variable"] and "at_risk" not in fields["algorithm"]     # vague: clarify instead


def test_with_the_switch_off_the_plan_has_no_next_questions(monkeypatch):
    monkeypatch.setenv("C1_ACTIVE", "off")
    s = plan_store(algorithm=("Neural Network", False))
    p = plan_from_events("t", list(s.events))
    assert p["next_questions"] == [] and not any("needs_detail" in f for f in p["fields"])
