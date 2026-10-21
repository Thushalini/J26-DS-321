import re

import pytest

from component4.common import config
from component4.m1_feature_integration import project_health
from component4.m4_readiness.rubric import assess

REPO = ["README.md", "requirements.txt", "params.yaml", "data/raw.csv.dvc", "src/train.py", "notebooks/eda.ipynb",
        "tests/test_train.py", ".github/workflows/ci.yml", "Dockerfile", "dvc.yaml", "schemas/input.json",
        "models/model.pkl", "metrics.json", "monitoring/drift.py", "docs/index.md", "configs/train.yaml"]


def test_reproducibility_counts_its_own_evidence():
    assert project_health.reproducibility(REPO) == 1.0
    assert project_health.reproducibility(["notebooks/eda.ipynb"]) == 0.0
    assert project_health.reproducibility([]) is None


def test_no_file_counts_for_both_readiness_and_project_health():
    readiness = [p for area in config.READINESS["areas"].values() for c in area["criteria"].values() for p in c["patterns"]]
    health = [p for c in config.PROJECT_HEALTH["reproducibility"].values() for p in c["patterns"]]
    health += config.PROJECT_HEALTH["data_quality"]["validation_patterns"]
    for path in REPO:
        lowered = path.lower()
        in_readiness = any(re.search(p, lowered) for p in readiness)
        in_health = any(re.search(p, lowered) for p in health)
        assert not (in_readiness and in_health), path


def test_data_quality_from_leakage_with_a_small_validation_bonus():
    runs = [{"root_cause_probabilities": {"data_leakage": 0.2}}, {"root_cause_probabilities": {"data_leakage": 0.4}}]
    assert project_health.data_quality(runs, ["notebooks/a.ipynb"]) == 0.7
    assert project_health.data_quality(runs, ["schemas/input.json"]) == 0.8
    assert project_health.data_quality([{"root_cause_probabilities": {}}], REPO) is None


@pytest.mark.parametrize("text, expected", [
    ("F1 >= 0.85", 0.85), ("accuracy 85%", 0.85), ("macro F1 of .9", 0.9), ("F1", 0.80),
    ("top-5 accuracy", 0.80), (None, 0.80), ("RMSE below 3", None), ("log loss", None),
])
def test_target_comes_from_the_blueprint_metric_or_the_default(text, expected):
    assert project_health.target(text) == expected


def test_model_performance_is_best_score_over_target_capped_at_one():
    runs = [{"final_val_accuracy": 0.68}, {"final_val_accuracy": 0.72}, {}]
    assert project_health.model_performance(runs, "accuracy >= 0.80") == 0.9
    assert project_health.model_performance(runs, "accuracy >= 0.70") == 1.0
    assert project_health.model_performance([{}], "F1") is None
    assert project_health.model_performance(runs, "RMSE") is None


def test_readiness_still_scores_four_areas_without_the_moved_items():
    result = assess(REPO)
    assert {a["id"] for a in result["areas"]} == {"versioning", "monitoring", "pipeline", "documentation"}
    assert all(a["score"] > 0 for a in result["areas"])
