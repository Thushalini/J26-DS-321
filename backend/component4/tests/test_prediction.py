import dataclasses

import pytest

from component4.common import config
from component4.m2_prediction import data, models
from component4.m2_prediction.metrics import evaluate
from component4.m2_prediction.train import train_and_evaluate
from component4.synthetic import feature_spec, generator


@pytest.fixture(scope="module")
def small_run():
    return train_and_evaluate(generator.generate(500), search=False)


def test_split_is_stratified_and_complete():
    frame = generator.generate(1000)
    parts = data.split(frame)
    sizes = {name: len(y) for name, (_, y) in parts.items()}
    assert sum(sizes.values()) == len(frame)
    rates = [y.mean() for _, y in parts.values()]
    assert max(rates) - min(rates) < 0.03


def test_prediction_refuses_a_marks_feature(monkeypatch):
    marks = dataclasses.replace(feature_spec.FEATURES[0], name="gpa", source="academic_marks")
    monkeypatch.setattr(data, "FEATURES", [*feature_spec.FEATURES, marks])
    with pytest.raises(ValueError):
        data.prediction_features()


def test_every_model_is_reported(small_run):
    report, _ = small_run
    expected = {"rule_based", "logistic_regression", "random_forest", "gradient_boosting",
                "stacking_uncalibrated", "stacking_calibrated"}
    assert set(report["test_metrics"]) == expected
    assert report["calibration"]["chosen"] in config.CALIBRATION["methods"]


def test_calibrated_model_returns_probabilities(small_run):
    _, model = small_run
    x = generator.generate(50, seed=99)[data.prediction_features()]
    proba = model.predict_proba(x)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_rule_baseline_flags_only_projects_with_all_three_signs():
    rule = config.RULE_BASELINE
    frame = generator.generate(10)[data.prediction_features()].copy()
    frame.loc[:, "drift_score"] = rule["drift_score_at_least"]
    frame.loc[:, "issue_count"] = rule["issue_count_at_least"]
    frame.loc[:, "reproducibility"] = rule["reproducibility_at_most"]
    frame.loc[0, "reproducibility"] = 1.0
    proba = models.rule_baseline_proba(frame)
    assert proba[0] == 1.0 and (proba[1:] == 0.0).all()


def test_perfect_predictions_score_perfectly():
    metrics = evaluate([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
    assert metrics["auc_roc"] == 1.0 and metrics["brier"] == 0.0 and metrics["macro_f1"] == 1.0


def test_tuning_and_cross_validation_are_reported():
    report, _ = train_and_evaluate(generator.generate(400), search=True)
    assert set(report["tuning"]) == {"logistic_regression", "random_forest", "gradient_boosting"}
    assert set(report["cross_validation"]) == {"logistic_regression", "random_forest", "gradient_boosting", "stacking"}
    for scores in report["cross_validation"].values():
        assert set(scores) == set(config.CROSS_VALIDATION["scoring"])
        assert 0 <= scores["brier"]["mean"] <= 1


def test_a_near_tie_keeps_the_preferred_calibration(monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from component4.common import config as settings
    from component4.m2_prediction import models
    from component4.m2_prediction.data import split as split_data
    from component4.synthetic import generator as gen
    parts = split_data(gen.generate(400))
    monkeypatch.setitem(settings.CALIBRATION, "switch_margin", 1.0)
    _, chosen, _ = models.calibrate_best(LogisticRegression(max_iter=500), *parts["train"], *parts["val"])
    assert chosen == settings.CALIBRATION["preferred"]
