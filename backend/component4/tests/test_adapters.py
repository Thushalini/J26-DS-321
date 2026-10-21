import pytest

from component4.m1_feature_integration import assemble, c1_adapter, c2_adapter
from component4.synthetic.feature_spec import feature_names


def blueprint(novelty=0.6, feasibility=None, statuses=("confirmed", "known", "missing", "conflicting")):
    return {"version": 3, "novelty_score": novelty, "feasibility": feasibility,
            "fields": {f"f{i}": {"status": s} for i, s in enumerate(statuses)}}


def summary(scores):
    return {"lagging_stages": ["model_evaluation"], "stage_breakdown": {"eda": 0.4},
            "drift": {"overall_severity": "moderate",
                      "dimensions": [{"dimension": d, "drift_score": s} for d, s in scores.items()]}}


def roles(*members, chat_weight=0.5):
    return {"chat_weight": chat_weight, "members": [
        {"member_id": f"M-{i}", "code_items": c, "chat_items": h, "evidence": "none" if c + h == 0 else "ok"}
        for i, (c, h) in enumerate(members)]}


def test_feasibility_checks_become_one_score():
    assert c1_adapter.feasibility_score({"data": "pass", "compute": "warning", "time": "fail"}) == 0.5
    assert c1_adapter.feasibility_score(None) is None


def test_c1_features_and_settled_fields():
    out = c1_adapter.project_features(blueprint(feasibility={"a": "pass"}))
    assert out["features"] == {"novelty_score": 0.6, "feasibility_score": 1.0}
    assert (out["context"]["fields_settled"], out["context"]["fields_total"]) == (2, 4)


def test_c1_missing_scores_stay_missing():
    assert c1_adapter.project_features(blueprint(novelty=None))["features"] == {"novelty_score": None, "feasibility_score": None}


def test_drift_is_the_worst_dimension_and_names_the_driver():
    out = c2_adapter.project_features(summary({"target": 0.1, "algorithm": 0.7, "metric": 0.3}))
    assert out["features"]["drift_score"] == 0.7 and out["context"]["drift_driver"] == "algorithm"


def test_drift_ignores_missing_dimension_scores():
    assert c2_adapter.overall_drift(summary({"target": None, "metric": 0.2})) == (0.2, "metric")
    assert c2_adapter.overall_drift(summary({"target": None})) == (None, None)


def test_even_work_is_balanced_and_one_person_is_not():
    assert c2_adapter.contribution_balance(roles((10, 0), (10, 0), (10, 0))) == pytest.approx(1.0)
    assert c2_adapter.contribution_balance(roles((30, 0), (0, 0), (0, 0))) < 0.4


def test_chat_counts_at_component2_weight():
    assert c2_adapter.contribution_balance(roles((10, 0), (0, 20), chat_weight=0.5)) == pytest.approx(1.0)


def test_inactive_members_are_reported():
    assert c2_adapter.project_features(summary({}), roles((5, 0), (0, 0)))["context"]["inactive_members"] == ["M-1"]


def test_assemble_lists_every_feature_and_what_is_missing():
    out = assemble.assemble(blueprint=blueprint(feasibility={"a": "pass"}),
                            c2_summary=summary({"target": 0.3}), c2_roles=roles((5, 1), (4, 2)),
                            c3_runs=[{"severity_level": "HIGH", "severity_score": 0.9, "anomaly_score": 0.8}],
                            project_health={"timeline_adherence": 0.7, "not_a_feature": 1})
    assert list(out["features"]) == feature_names()
    assert set(out["missing"]) == {"model_performance", "data_quality", "reproducibility", "mlops_readiness_score"}
    assert "not_a_feature" not in out["features"]
