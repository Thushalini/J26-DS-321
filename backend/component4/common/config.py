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

TUNING = {
    "cv_folds": 5,
    "scoring": "neg_brier_score",
    "grids": {
        "logistic_regression": {"logisticregression__C": [0.1, 1.0, 10.0]},
        "random_forest": {"min_samples_leaf": [2, 5, 10], "max_depth": [None, 8]},
        "gradient_boosting": {"n_estimators": [100, 200], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]},
    },
}

CROSS_VALIDATION = {
    "folds": 5,
    "scoring": {"auc_roc": "roc_auc", "brier": "neg_brier_score", "macro_f1": "f1_macro"},
}

SEED_STABILITY = {
    "seeds": [42, 7, 123, 2024, 99],
    "metrics": ["auc_roc", "brier", "ece", "macro_f1"],
}
SEED_STABILITY_JSON = RESULTS_DIR / "seed_stability.json"

FORECAST_EXAMPLE_JSON = RESULTS_DIR / "forecast_example.json"

FORECAST = {
    "weeks": 8,
    "simulations": 500,
    "percentiles": [10, 25, 50, 75, 90],
    "weekly_change": {"unit": 0.04, "score100": 4.0},
    "weekly_new_issues": 0.3,
    "weekly_new_high_issues": 0.1,
    "weekly_resolved_high_share": 0.3,
}

C3_RESULTS_DIR = BASE_DIR.parent / "component3" / "results"
FAILURE_PATTERNS_JSON = RESULTS_DIR / "failure_patterns.json"

FAILURE_PATTERNS = {
    "causes": ["overfitting", "stalled", "data_leakage", "resource_bottleneck", "premature_evaluation"],
    "k_range": [2, 3, 4, 5, 6, 7, 8],
    "cause_present_at": 0.5,
    "recurring_at_least": 2,
    "n_init": 10,
}

RECOMMENDATIONS = {
    "text": {"weak": "{label}: lowering the probability by {points:.1f} pts",
             "pattern": "recurring issue pattern: {cause} ({count} {unit})", "issue": "issue", "issues": "issues"},
    "top_n": 3,
    "weak_at_points": -1.0,
    "min_gain_points": 0.5,
    "actions": {
        "realign_plan": {
            "title": "Re-align the work with the agreed target, algorithm and metric",
            "stage": "model_development",
            "changes": {"drift_score": -0.2},
            "when_weak": ["drift_score"],
            "when_pattern": [],
        },
        "resolve_high_issues": {
            "title": "Resolve the open high-severity experiment issues",
            "stage": "model_evaluation",
            "changes": {"open_high_issues": "zero", "max_severity": -0.2},
            "when_weak": ["open_high_issues", "max_severity"],
            "when_pattern": [],
        },
        "improve_reproducibility": {
            "title": "Fix seeds, configs and environments so every experiment can be re-run",
            "stage": "deployment_mlops",
            "changes": {"reproducibility": 0.2},
            "when_weak": ["reproducibility"],
            "when_pattern": [],
        },
        "improve_data_quality": {
            "title": "Clean the data and check for leakage before training",
            "stage": "data_cleaning",
            "changes": {"data_quality": 0.2, "issue_share": -0.1},
            "when_weak": ["data_quality"],
            "when_pattern": ["data_leakage"],
        },
        "rebalance_work": {
            "title": "Rebalance the work so no member or stage is left behind",
            "stage": None,
            "changes": {"contribution_balance": 0.2},
            "when_weak": ["contribution_balance"],
            "when_pattern": [],
        },
        "replan_timeline": {
            "title": "Re-plan the remaining milestones and trim scope where needed",
            "stage": "documentation",
            "changes": {"timeline_adherence": 0.15},
            "when_weak": ["timeline_adherence"],
            "when_pattern": [],
        },
        "improve_model": {
            "title": "Regularise or rethink the model to close the gap to the target metric",
            "stage": "model_development",
            "changes": {"model_performance": 0.15, "issue_share": -0.1},
            "when_weak": ["model_performance"],
            "when_pattern": ["overfitting", "stalled"],
        },
        "evaluate_properly": {
            "title": "Train long enough and log tuning before judging a model",
            "stage": "model_evaluation",
            "changes": {"issue_share": -0.1},
            "when_weak": ["issue_share"],
            "when_pattern": ["premature_evaluation"],
        },
        "fix_compute": {
            "title": "Fix the compute bottleneck: batch size, data loading or a larger instance",
            "stage": "deployment_mlops",
            "changes": {"issue_share": -0.1},
            "when_weak": [],
            "when_pattern": ["resource_bottleneck"],
        },
        "mlops_basics": {
            "title": "Add model versioning and basic monitoring",
            "stage": "deployment_mlops",
            "changes": {"mlops_readiness_score": 15},
            "when_weak": ["mlops_readiness_score"],
            "when_pattern": [],
        },
    },
}

IMPROVEMENT_SCENARIO_JSON = RESULTS_DIR / "improvement_scenario.json"

IMPROVEMENT_SCENARIO = {
    "starting_changes": {"reproducibility": 0.15, "issue_count": 8, "open_high_issues": 3, "issue_share": 0.7},
    "recurring": [{"causes": ["premature_evaluation"], "issues": 4}],
}

C1 = {
    "feasibility_points": {"pass": 1.0, "warning": 0.5, "fail": 0.0},
    "settled_field_statuses": ["known", "confirmed"],
}

C2 = {
    "drift_dimensions": ["target", "algorithm", "metric"],
    "default_chat_weight": 0.5,
    "inactive_evidence": "none",
}

FEATURE_LABELS = {
    "novelty_score": "Novelty of the approach",
    "feasibility_score": "Feasibility of the plan",
    "drift_score": "Drift from the blueprint",
    "contribution_balance": "Balance of contributions",
    "issue_count": "Experiment issues",
    "issue_share": "Share of runs with issues",
    "max_severity": "Worst issue severity",
    "mean_severity": "Average issue severity",
    "open_high_issues": "Open high-severity issues",
    "mean_anomaly_score": "Experiment anomaly level",
    "model_performance": "Model performance vs target",
    "data_quality": "Data quality",
    "reproducibility": "Reproducibility",
    "timeline_adherence": "Timeline adherence",
    "mlops_readiness_score": "MLOps readiness",
}

KIND_TEXT = {"unit": "on a 0-1 scale", "score100": "out of 100", "count": "in total"}

STAGE_LABELS = {
    "data_cleaning": "Data cleaning",
    "eda": "Exploratory analysis",
    "feature_engineering": "Feature engineering",
    "model_development": "Model development",
    "model_evaluation": "Model evaluation",
    "deployment_mlops": "Deployment and MLOps",
    "documentation": "Documentation",
}

API = {
    "prefix": "/c4",
    "forecast_simulations": 200,
    "demo_project_index": 0,
    "background_size": 50,
    "min_team_signals": 5,
}

HISTORY = {
    "points": 8,
    "changed_at_least": {"unit": 0.05, "count": 1, "score100": 5},
}

READINESS = {
    "status": "draft",
    "max_score": 100,
    "weak_below": 40,
    "text": {"found": "Found: {items}.", "missing": "Not found in the repository: {items}."},
    "areas": {
        "versioning": {
            "label": "Model versioning",
            "stage": "deployment_mlops",
            "criteria": {
                "dependencies": {"text": "dependencies listed", "points": 1, "patterns": [
                    r"(^|/)requirements[^/]*\.txt$", r"(^|/)environment\.ya?ml$", r"(^|/)pyproject\.toml$",
                    r"(^|/)(pipfile|poetry)\.lock$", r"(^|/)setup\.py$"]},
                "data_versioning": {"text": "data or models versioned (DVC or similar)", "points": 1, "patterns": [
                    r"\.dvc$", r"(^|/)dvc\.(yaml|lock)$", r"(^|/)\.dvc/"]},
                "experiment_tracking": {"text": "experiments tracked (MLflow, W&B or similar)", "points": 1, "patterns": [
                    r"(^|/)mlruns/", r"(^|/)mlproject$", r"mlflow", r"(^|/)wandb/", r"(^|/)\.neptune/"]},
            },
        },
        "monitoring": {
            "label": "Monitoring and drift detection",
            "stage": "deployment_mlops",
            "criteria": {
                "monitoring": {"text": "monitoring or drift checks", "points": 1, "patterns": [
                    r"monitor", r"drift", r"evidently", r"whylogs"]},
                "data_validation": {"text": "data validation", "points": 1, "patterns": [
                    r"data_?validation", r"validate_?data", r"great_expectations", r"pandera",
                    r"(^|/)expectations/", r"(^|/)schemas?/"]},
                "alerts_logging": {"text": "alerts or logging set up", "points": 1, "patterns": [
                    r"prometheus", r"grafana", r"alertmanager", r"(^|/)alerts?\.(py|ya?ml)$",
                    r"(^|/)logging\.(conf|ini|ya?ml|json)$"]},
            },
        },
        "pipeline": {
            "label": "Pipeline robustness",
            "stage": "deployment_mlops",
            "criteria": {
                "tests": {"text": "automated tests", "points": 1, "patterns": [
                    r"(^|/)tests?/", r"(^|/)test_[^/]*\.py$", r"_test\.py$", r"(^|/)conftest\.py$", r"(^|/)pytest\.ini$"]},
                "ci": {"text": "continuous integration", "points": 1, "patterns": [
                    r"(^|/)\.github/workflows/[^/]+\.ya?ml$", r"(^|/)\.gitlab-ci\.yml$", r"(^|/)jenkinsfile$",
                    r"(^|/)azure-pipelines\.ya?ml$", r"(^|/)\.circleci/"]},
                "scripted_pipeline": {"text": "scripted pipeline or config files", "points": 1, "patterns": [
                    r"(^|/)dvc\.yaml$", r"(^|/)makefile$", r"(^|/)pipelines?/", r"(^|/)pipeline[^/]*\.py$",
                    r"(^|/)dags/", r"(^|/)params\.ya?ml$", r"(^|/)configs?/[^/]+\.(ya?ml|json|toml)$"]},
                "container": {"text": "container setup", "points": 1, "patterns": [
                    r"(^|/)dockerfile", r"(^|/)(docker-)?compose\.ya?ml$", r"(^|/)\.devcontainer/"]},
            },
        },
        "documentation": {
            "label": "Documentation",
            "stage": "documentation",
            "criteria": {
                "readme": {"text": "README", "points": 1, "patterns": [r"(^|/)readme(\.[a-z]+)?$"]},
                "docs": {"text": "docs folder", "points": 1, "patterns": [r"(^|/)docs?/", r"(^|/)mkdocs\.ya?ml$"]},
                "model_card": {"text": "model or data card", "points": 1, "patterns": [
                    r"model[_-]?card", r"data[_-]?card", r"datasheet"]},
            },
        },
    },
}

COMPETENCY = {
    "module_stages_file": BASE_DIR / "m6_improvement" / "competency" / "module_stages.json",
    "grade_points": {
        "A+": 4.0, "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
        "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "E": 0.0, "F": 0.0,
    },
    "max_points": 4.0,
    "min_evidence": 2.0,
    "max_sheets": 10,
    "max_sheet_bytes": 2_000_000,
    "group_roles": ["lead", "member", "supervisor"],
    "member_roles": ["lead", "member"],
}

CAUSE_STAGES = {
    "overfitting": ["model_development", "model_evaluation"],
    "stalled": ["model_development"],
    "data_leakage": ["data_cleaning", "feature_engineering"],
    "resource_bottleneck": ["deployment_mlops"],
    "premature_evaluation": ["model_evaluation"],
}

TEAM_SOURCES = {
    "c2_window_days": 30,
    "c2_roles_window": "30d",
    "min_stage_items": 20,
}
