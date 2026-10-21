"""Evaluation helpers: SUS scoring (Brooke, 1996) and reading stages from the self-report form."""
import pytest

from component2.evaluation.self_report_check import to_stage
from component2.evaluation.sus_score import sus


@pytest.mark.parametrize("answers, score", [
    ([5, 1] * 5, 100.0),        # agrees with every positive item, disagrees with every negative one
    ([1, 5] * 5, 0.0),
    ([3] * 10, 50.0),
    ([4, 2, 4, 2, 4, 2, 4, 2, 4, 2], 75.0),
])
def test_sus_score(answers, score):
    assert sus(answers) == score


@pytest.mark.parametrize("text, stage", [
    ("Model development", "model_development"),
    (" EDA ", "eda"),
    ("Exploratory data analysis", "eda"),
    ("Deployment / MLOps", "deployment_mlops"),
    ("feature engineering", "feature_engineering"),
    ("documentation", "documentation"),
    ("", None),
    (None, None),
    ("cooking", None),
])
def test_self_report_stage_names(text, stage):
    assert to_stage(text) == stage
