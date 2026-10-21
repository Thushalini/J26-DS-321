import numpy as np
import pytest

from component4.m2_prediction.data import split
from component4.m2_prediction.train import train_and_evaluate
from component4.m3_explainability import checks
from component4.m3_explainability.shap_explainer import global_importance, local_explanation, shap_values
from component4.synthetic import generator
from component4.synthetic.feature_spec import feature_names


@pytest.fixture(scope="module")
def explained():
    frame = generator.generate(500)
    _, model = train_and_evaluate(frame, search=False)
    parts = split(frame)
    rows = parts["test"][0].iloc[:5]
    return model, rows, shap_values(model, parts["train"][0].iloc[:20], rows)


def test_contributions_add_up_to_the_prediction(explained):
    model, rows, explanation = explained
    rebuilt = explanation.base_values + explanation.values.sum(axis=1)
    assert np.allclose(rebuilt, model.predict_proba(rows)[:, 1], atol=1e-6)


def test_local_explanation_is_sorted_by_size(explained):
    _, _, explanation = explained
    sizes = [abs(f["contribution_points"]) for f in local_explanation(explanation, 0)["factors"]]
    assert sizes == sorted(sizes, reverse=True)


def test_global_importance_covers_every_feature(explained):
    _, _, explanation = explained
    assert {row["feature"] for row in global_importance(explanation)} == set(feature_names())


def test_checks_report_every_feature(explained):
    model, rows, explanation = explained
    assert [r["feature"] for r in checks.direction_check(explanation)] == feature_names()
    assert [r["feature"] for r in checks.sensitivity_analysis(model, rows)] == feature_names()


def test_tree_models_follow_each_signals_direction():
    from component4.m2_prediction.models import baselines
    from component4.synthetic.feature_spec import FEATURES
    parts = split(generator.generate(500))
    x, y = parts["train"]
    rows = parts["test"][0].iloc[:20]
    for name in ("random_forest", "gradient_boosting"):
        model = baselines()[name].fit(x, y)
        before = model.predict_proba(rows)[:, 1]
        for spec in FEATURES:
            nudged = rows.copy()
            nudged[spec.name] = nudged[spec.name] + spec.direction * nudged[spec.name].abs().clip(lower=1) * 0.2
            assert (model.predict_proba(nudged)[:, 1] >= before - 1e-9).all(), (name, spec.name)
