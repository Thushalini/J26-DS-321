"""Drift detector: how far the team's recent work is from its blueprint (MiniLM replaced by fake_embed)."""
import json
from collections import Counter

import pytest

from component2.drift import detect_drift
from component2.drift.detect_drift import score_dimension, severity_of, texts_of

TEAM = "demo__drift"


@pytest.fixture(autouse=True)
def no_model(monkeypatch, fake_embed):
    monkeypatch.setattr(detect_drift, "embed", fake_embed)


@pytest.mark.parametrize("score, severity", [
    (0.0, "none"), (0.19, "none"), (0.20, "minor"), (0.39, "minor"),
    (0.40, "moderate"), (0.59, "moderate"), (0.60, "significant"), (1.0, "significant"),
])
def test_severity_limits(score, severity):
    assert severity_of(score) == severity


def test_no_plan_and_no_evidence_have_no_score():
    assert score_dimension("algorithm", None, Counter({"xgboost": 2}))[:2] == (None, "no_plan")
    assert score_dimension("algorithm", "random forest", Counter())[:2] == (None, "no_evidence")


def test_all_mentions_on_plan():
    score, severity, top, why = score_dimension("algorithm", "Random Forest", Counter({"random forest": 4}))
    assert (score, severity, top) == (0.0, "none", "random forest")
    assert "All 4 mentions" in why


def test_share_of_off_plan_mentions_is_the_score():
    score, severity, top, why = score_dimension("algorithm", "random forest",
                                                Counter({"random forest": 1, "xgboost": 3}))
    assert (score, severity, top) == (0.75, "significant", "xgboost")
    assert "'xgboost' (3x)" in why


def test_similar_meaning_counts_as_on_plan():
    # fake_embed: identical words -> similarity 1.0, above MATCH_AT, so it is not drift
    score, _, _, _ = score_dimension("target", "Customer churn", Counter({"customer churn": 2}))
    assert score == 0.0


def test_code_files_of_one_commit_count_once():
    rows = [("t", "Train RF", "commit", "{}"),
            ("t", "rf = RandomForest()", "code_change", json.dumps({"commit": "abc"})),
            ("t", "RandomForest(n=10)", "code_change", json.dumps({"commit": "abc"}))]
    texts = texts_of(rows)
    assert texts == ["Train RF", "rf = RandomForest()\nRandomForest(n=10)"]


def work(add):
    add(TEAM, [("c1", "M-aaaaaa", "commit", "2026-08-01T10:00:00Z", "Tried SVM first", None),
               ("c2", "M-aaaaaa", "commit", "2026-10-01T10:00:00Z", "Train random forest, accuracy 0.8", None),
               ("c3", "M-bbbbbb", "commit", "2026-10-10T10:00:00Z", "Switched to xgboost", None),
               ("c4", "M-bbbbbb", "commit", "2026-10-20T10:00:00Z", "Check accuracy again", None)])


def test_detect_team_against_the_apps_blueprint(conn, add):
    work(add)
    plan = {"target": None, "algorithm": "random forest", "metric": "accuracy"}
    results = {r["dimension"]: r for r in detect_drift.detect_team(conn, TEAM, 30, planned=plan)}
    assert results["target"]["severity"] == "no_plan"
    assert results["algorithm"]["drift_score"] == 0.5          # random forest 1, xgboost 1; SVM is too old
    assert results["metric"]["drift_score"] == 0.0
    assert {r["blueprint"] for r in results.values()} == {"app"}
    assert {r["window_end"] for r in results.values()} == {"2026-10-20T10:00:00Z"}


def test_blueprint_is_inferred_from_early_work_and_saved(conn, add, tmp_path, monkeypatch):
    monkeypatch.setattr(detect_drift, "BLUEPRINT_DIR", tmp_path)
    work(add)
    add(TEAM, [("e1", "M-aaaaaa", "commit", "2026-08-02T10:00:00Z", "SVM tuned", None),
               ("e2", "M-bbbbbb", "commit", "2026-10-11T10:00:00Z", "xgboost tuned", None),
               ("e3", "M-bbbbbb", "commit", "2026-10-12T10:00:00Z", "xgboost again", None)])
    results = {r["dimension"]: r for r in detect_drift.detect_team(conn, TEAM, 30)}
    # first 40% of 7 commits = the two SVM commits; over the whole timeline xgboost (3) would win
    assert results["algorithm"]["planned"] == "svm"
    assert results["algorithm"]["drift_score"] == 1.0
    saved = json.loads((tmp_path / f"{TEAM}.json").read_text(encoding="utf-8"))
    assert saved["_inferred_from_early_activity"] and saved["algorithm"]["name"] == "svm"


def test_team_without_activity_has_no_results(conn):
    assert detect_drift.detect_team(conn, "nobody", 30, planned={}) == []
