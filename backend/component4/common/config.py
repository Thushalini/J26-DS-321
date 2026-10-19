"""Paths and settings for Component 4. Every threshold and model setting is added here, not inline."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MARKS_DIR = DATA_DIR / "marks"

SYNTHETIC_CSV = DATA_DIR / "synthetic_projects.csv"
DATA_VALIDATION_JSON = RESULTS_DIR / "data_validation.json"
METRICS_JSON = RESULTS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "success_model.joblib"
MODEL_VERSION = "0.1-provisional"

SEED = 42

MODULE_PACKAGES = [
    "m1_feature_integration",
    "m2_prediction",
    "m3_explainability",
    "m4_readiness",
    "m5_forecast",
    "m6_improvement",
]

ID_COLUMN = "project_id"
OUTCOME_COLUMN = "outcome"
OUTCOMES = ["success", "delayed", "failed"]
POSITIVE_OUTCOMES = ["success"]

FORBIDDEN_FEATURE_SOURCES = {"academic_marks"}

SYNTHETIC = {
    "n_projects": 2000,
    "outcome_shares": {"success": 0.35, "delayed": 0.45, "failed": 0.20},
    "outcome_noise_scale": 0.6,
}

DATA_CHECKS = {
    "share_tolerance": 0.02,
    "min_abs_correlation": 0.05,
}

SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

MODEL_PARAMS = {
    "logistic_regression": {"max_iter": 1000},
    "random_forest": {"n_estimators": 300, "min_samples_leaf": 5},
    "gradient_boosting": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    "stacking_cv": 5,
}

CALIBRATION = {
    "methods": ["sigmoid", "isotonic"],
    "cv": 5,
    "bins": 10,
}

DECISION_THRESHOLD = 0.5

RULE_BASELINE = {
    "drift_score_at_least": 0.4,
    "issue_count_at_least": 3,
    "reproducibility_at_most": 0.4,
}

EXPLAINABILITY_JSON = RESULTS_DIR / "explainability.json"

EXPLAINABILITY = {
    "background_size": 50,
    "explain_size": 100,
    "min_direction_correlation": 0.1,
    "probability_to_points": 100,
}

SENSITIVITY = {
    "steps": {"unit": 0.1, "score100": 10, "count": 1},
    "min_share_in_expected_direction": 0.9,
    "sample_size": 300,
}

C3 = {
    "issue_levels": ["MEDIUM", "HIGH"],
    "high_level": "HIGH",
    "recent_window_days": 14,
}
