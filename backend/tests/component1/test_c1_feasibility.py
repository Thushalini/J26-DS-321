"""Tests for the dataset-size field and the feasibility classifier (simulate, train, predict, checks).

A small model is trained here from the simulator, so the tests do not depend on the saved model.
All plans and messages are invented; fixed dates keep the tests independent of today.
"""
import csv
from copy import deepcopy
from datetime import date

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from component1.agent.policy import gap_text
from component1.agent.signals import GAP, Signal
from component1.checks.plan_checks import model_check, record_checks, run_checks
from component1.common.events import EventStore
from component1.common.schema import FieldName, FieldStatus, Provenance, SourceType
from component1.extraction.extractor import extract_items
from component1.extraction.rules import rule_items
from component1.extraction.size import find_sizes, rows_of, standard_size
from component1.feasibility import expert_sheet
from component1.feasibility.predict import blueprint_inputs, judge, plan_features_from_values
from component1.feasibility.simulate import (
    ASSUMPTIONS, FEATURE_NAMES, LABELS, features, generate_dataset, outcome, rule_baseline,
)
from component1.feasibility.train import best_single_feature, read_expert, train_and_evaluate
from component1.plan import plan_from_events


def small_model():
    X, y = generate_dataset(4000, seed=3)
    model = RandomForestClassifier(n_estimators=80, min_samples_leaf=5, random_state=0).fit(X, y)
    return {"model": model, "labels": LABELS, "features": FEATURE_NAMES}


BUNDLE = small_model()


def plan(**values):
    store = EventStore("t")
    for field, value in values.items():
        prov = Provenance(source_type=SourceType.CHAT, source_id="m", speaker="M-1", text_span=value)
        store.update_field(FieldName(field), value, FieldStatus.KNOWN, 0.5, prov, "extractor")
    return store


# ------------------------------------------------------------ dataset size
@pytest.mark.parametrize("text, rows", [
    ("The dataset has about 12k rows.", 12000), ("We have 2 million records.", 2_000_000),
    ("It's 5,000 samples with 25 features.", 5000), ("1.5M rows of transactions", 1_500_000),
    ("around fifty thousand rows", 50_000), ("We only have 800 labelled examples.", 800),
    ("The dataset has 10 lakh rows", 1_000_000),
])
def test_dataset_sizes_are_found_and_counted_in_rows(text, rows):
    assert [r for _, r in find_sizes(text)] == [rows]
    assert [(i.value, i.span in text) for i in rule_items(text) if i.field == FieldName.DATASET_SIZE] == \
        [(f"{rows} rows", True)]


@pytest.mark.parametrize("text", ["500 customers visited", "3 people joined", "a few rows", "We need 12k more."])
def test_numbers_without_a_row_word_are_not_a_dataset_size(text):
    assert find_sizes(text) == []


def test_sizes_are_stored_in_one_standard_form():
    assert standard_size("12k rows") == standard_size("12,000 records") == "12000 rows"
    assert rows_of("12000 rows") == 12000 and rows_of("~3.5 million") == 3_500_000 and rows_of("lots") is None


def test_the_language_model_s_sizes_are_standardised_too():
    reply = '{"items": [{"field": "dataset_size", "value": "about 12,000 records", "span": "12,000 records"}]}'
    (item,) = extract_items("It has 12,000 records.", lambda prompt: reply)
    assert item.value == "12000 rows"


def test_buddy_asks_for_the_size_in_plain_words():
    assert "how many rows" in gap_text(Signal(GAP, "dataset_size"))


# --------------------------------------------------------------- simulator
def test_the_simulated_data_is_balanced_enough_and_reproducible():
    X, y = generate_dataset(3000, seed=1)
    shares = {label: float(np.mean(y == label)) for label in LABELS}
    assert all(share > 0.15 for share in shares.values())
    X2, y2 = generate_dataset(3000, seed=1)
    assert (X == X2).all() and (y == y2).all() and X.shape[1] == len(FEATURE_NAMES)


def quiet():
    """The assumptions without any noise, so one change has one clear effect."""
    a = deepcopy(ASSUMPTIONS)
    a.update(team_skill_sigma=0.0, available_time_sigma=0.0, data_quality_sigma=0.0)
    return a


def test_outcomes_follow_the_assumptions():
    rng, a = np.random.default_rng(0), quiet()
    assert outcome(rng, "deep", 2000, False, 10, a)[0] == "infeasible"           # far too little data
    assert outcome(rng, "tree", 50_000, False, 10, a)[0] == "feasible"
    slow = outcome(rng, "deep", 200_000, False, 8, a)[1]["hours_needed"]
    fast = outcome(rng, "deep", 200_000, True, 8, a)[1]["hours_needed"]
    assert fast < slow                                                              # a GPU helps deep models
    assert outcome(rng, "tree", 50_000, False, 20, a)[1]["time_ratio"] < \
        outcome(rng, "tree", 50_000, False, 2, a)[1]["time_ratio"]                  # more weeks, more room


def test_the_rule_baseline_uses_fixed_limits():
    def row(family, rows, gpu, weeks):
        return features("classification", family, rows, gpu, weeks)
    out = rule_baseline(np.array([row("tree", 300, False, 10), row("deep", 50_000, False, 10),
                                  row("deep", 50_000, True, 2), row("tree", 1500, False, 10),
                                  row("tree", 50_000, False, 10)]))
    assert list(out) == ["infeasible", "risky", "risky", "risky", "feasible"]


# ---------------------------------------------------------------- training
def test_the_trained_model_beats_the_rules_and_one_feature_alone():
    r = train_and_evaluate(n=3000, n_estimators=40, save=False)
    assert r["random_forest"]["macro_f1"] > r["rule_baseline"]["macro_f1"] + 0.1
    assert r["random_forest"]["macro_f1"] > r["best_single_feature"]["macro_f1"] + 0.1
    assert r["data"]["kind"].startswith("SIMULATED") and "assumptions" in r["data"]


def test_best_single_feature_reports_a_feature_name():
    X, y = generate_dataset(800, seed=2)
    out = best_single_feature(X, y)
    assert out["feature"] in FEATURE_NAMES and 0 < out["macro_f1"] < 1


# -------------------------------------------------------------- predicting
def test_a_small_deep_plan_is_judged_unfeasible_with_reasons_and_better_options():
    j = judge(BUNDLE, "classification", "LSTM", 2000, has_gpu=False, weeks_left=3)
    assert j["label"] in ("risky", "infeasible") and abs(sum(j["probabilities"].values()) - 1) < 0.02
    assert j["reasons"] and all(r["effect"] == "raises risk" for r in j["reasons"])
    assert j["alternatives"]
    for a in j["alternatives"]:
        assert a["feasible_probability"] >= j["probabilities"]["feasible"] + 0.1
        assert "LSTM" not in a["change"]


def test_a_comfortable_plan_is_feasible_with_no_changes_suggested():
    j = judge(BUNDLE, "classification", "XGBoost", 50_000, has_gpu=False, weeks_left=12)
    assert j["label"] == "feasible" and j["alternatives"] == []
    assert all(r["effect"] == "lowers risk" for r in j["reasons"])


def test_plans_the_model_cannot_judge_return_none():
    assert judge(BUNDLE, "classification", "our own model", 5000) is None
    assert judge(BUNDLE, "classification", "K-Means", 5000) is None        # a mismatch is the rule checks' job
    assert judge(BUNDLE, "classification", "XGBoost", 0) is None


def test_the_task_can_come_from_the_algorithm():
    assert judge(BUNDLE, "", "ARIMA", 500, weeks_left=10)["inputs"]["task"] == "forecasting"


def test_plain_values_become_model_inputs():
    x = plan_features_from_values("regression", "random forest", "20k rows", gpu="yes", weeks_left="6")
    assert x == features("regression", "tree", 20000, True, 6.0)
    assert plan_features_from_values("regression", "?", "20k rows") is None


# --------------------------------------------------------- from a blueprint
def test_inputs_come_from_the_blueprint_and_need_algorithm_and_size():
    assert blueprint_inputs(plan(algorithm="XGBoost").state) is None
    store = plan(task_type="classification", algorithm="LSTM", dataset_size="2000 rows", resources="Google Colab",
                 timeline="deadline 2026-11-09")
    inputs = blueprint_inputs(store.state, store.events, today=date(2026, 10, 19))
    assert inputs == {"task": "classification", "algorithm": "LSTM", "rows": 2000, "has_gpu": True,
                      "weeks_left": 3.0}


def test_the_model_verdict_is_a_plan_check_stored_for_component_4():
    store = plan(task_type="classification", algorithm="LSTM", dataset_size="2000 rows")
    results = {r.key: r for r in record_checks(store, feasibility_bundle=BUNDLE)}
    check = results["model_feasibility"]
    assert check.status in ("warning", "fail") and check.field == "plan"
    assert check.detail.startswith("it looks ") and "Changes that would help" in check.detail
    assert store.state.feasibility["model_feasibility"] == check.status


def test_the_check_wording_does_not_change_from_day_to_day():
    store = plan(task_type="classification", algorithm="LSTM", dataset_size="2000 rows", timeline="deadline 2026-11-30")
    tuesday = {r.key: r for r in run_checks(store.state, store.events, date(2026, 11, 3), BUNDLE)}["model_feasibility"]
    wednesday = {r.key: r for r in run_checks(store.state, store.events, date(2026, 11, 4), BUNDLE)}["model_feasibility"]
    assert tuesday.detail == wednesday.detail               # 27 and 26 days left are both 3 whole weeks


def test_a_feasible_verdict_passes_and_says_nothing():
    j = judge(BUNDLE, "classification", "XGBoost", 50_000, weeks_left=12)
    assert model_check(j).status == "pass" and model_check(j).detail == ""


def test_without_a_model_the_check_is_skipped():
    store = plan(task_type="classification", algorithm="LSTM", dataset_size="2000 rows")
    assert "model_feasibility" not in {r.key for r in run_checks(store.state, store.events, feasibility_bundle=None)}


def test_the_plan_carries_the_verdict_or_none():
    assert plan_from_events("t", [])["feasibility_model"] is None
    store = plan(task_type="classification", algorithm="XGBoost", dataset_size="50000 rows")
    out = plan_from_events("t", store.events)
    assert out["feasibility_model"] is None or out["feasibility_model"]["label"] in LABELS
    size = next(f for f in out["fields"] if f["name"] == "dataset_size")
    assert size["rows"] == 50000


# ------------------------------------------------------------ expert sheet
def test_the_expert_sheet_covers_every_kind_of_plan_and_hides_the_answers(tmp_path):
    plans = expert_sheet.make_plans(40)
    assert len(plans) == 40 and all(p["label"] == "" for p in plans)
    assert {p["task"] for p in plans} == {"classification", "regression", "forecasting", "clustering"}
    path = tmp_path / "sheet.csv"
    expert_sheet.write_sheet(path, plans)
    with open(path, newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f).fieldnames) == expert_sheet.COLUMNS


def test_a_filled_in_expert_sheet_is_read_and_scored(tmp_path):
    plans = expert_sheet.make_plans(12)
    for p, label in zip(plans, ["feasible", "risky", "infeasible"] * 4):
        p["label"] = label
    plans[0]["label"] = ""                                   # not labelled yet: skipped
    path = tmp_path / "sheet.csv"
    expert_sheet.write_sheet(path, plans)
    X, y = read_expert(path)
    assert len(y) == 11 and X.shape[1] == len(FEATURE_NAMES)
    r = train_and_evaluate(n=1500, n_estimators=20, save=False, expert=path)
    assert r["expert_check"]["n"] == 11 and "macro_f1" in r["expert_check"]["random_forest"]
