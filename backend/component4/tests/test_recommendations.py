import pandas as pd
import pytest

from component4.common import config
from component4.m2_prediction.data import prediction_features, split
from component4.m2_prediction.train import train_and_evaluate
from component4.m6_improvement import recommendations as rec
from component4.synthetic import generator


@pytest.fixture(scope="module")
def setup():
    frame = generator.generate(500)
    _, model = train_and_evaluate(frame, search=False)
    return model, split(frame)["test"][0].iloc[0].astype(float)


def factor(name, points):
    return {"feature": name, "value": 0.0, "contribution_points": points}


def test_weak_signal_triggers_its_action(setup):
    model, project = setup
    project = project.copy()
    project["drift_score"] = 0.95
    ids = [r["id"] for r in rec.recommend(model, project, [factor("drift_score", -8.0)])]
    assert "realign_plan" in ids


def test_recurring_pattern_triggers_its_action(setup, monkeypatch):
    monkeypatch.setitem(rec.SETTINGS, "min_gain_points", float("-inf"))
    model, project = setup
    project = project.copy()
    project["data_quality"] = 0.1
    out = rec.recommend(model, project, [], recurring=[{"causes": ["data_leakage"], "issues": 5}])
    assert [r["id"] for r in out] == ["improve_data_quality"]
    assert "recurring issue pattern" in out[0]["reason"].lower()


def test_nothing_weak_means_no_recommendations(setup):
    model, project = setup
    assert rec.recommend(model, project, [factor("drift_score", 5.0)]) == []


def test_sorted_by_gain_and_capped(setup):
    model, project = setup
    weak = [factor(name, -5.0) for name in prediction_features()]
    out = rec.recommend(model, project, weak)
    gains = [r["expected_gain_points"] for r in out]
    assert gains == sorted(gains, reverse=True) and len(out) <= config.RECOMMENDATIONS["top_n"]


def test_changes_respect_ranges_and_consistency(setup):
    _, project = setup
    after = rec.apply_changes(project, {"reproducibility": 5.0, "issue_count": "zero"})
    assert after["reproducibility"] == 1.0 and after["issue_count"] == 0
    assert after["open_high_issues"] == 0 and after["issue_share"] == 0


def test_re_evaluation_reports_the_change(setup):
    model, project = setup
    after = rec.apply_changes(project, {"timeline_adherence": 0.3})
    result = rec.re_evaluate(model, project, after)
    assert list(result["changed_features"]) == ["timeline_adherence"]
    assert result["change_points"] == pytest.approx((result["probability_after"] - result["probability_before"]) * 100, abs=0.1)


def test_no_action_uses_academic_marks():
    for action in config.RECOMMENDATIONS["actions"].values():
        assert all(not name.startswith(("mark", "grade", "gpa")) for name in action["changes"])
