"""Explains the trained model: global importance, one example explanation, direction and sensitivity checks."""
from ..common import config
from . import checks
from .shap_explainer import global_importance, local_explanation, shap_values


def explain_model(model, x_background, x_explain, seed: int = config.SEED) -> dict:
    rows = x_explain.sample(min(len(x_explain), config.EXPLAINABILITY["explain_size"]), random_state=seed)
    explanation = shap_values(model, x_background, rows, seed)
    sensitivity_rows = x_explain.sample(min(len(x_explain), config.SENSITIVITY["sample_size"]), random_state=seed)

    direction = checks.direction_check(explanation)
    sensitivity = checks.sensitivity_analysis(model, sensitivity_rows)
    return {
        "explained_projects": len(rows),
        "global_importance": global_importance(explanation),
        "example_explanation": {"project_index": str(rows.index[0]), **local_explanation(explanation, 0)},
        "direction_check": direction,
        "sensitivity_analysis": sensitivity,
        "all_checks_passed": all(r["passed"] for r in direction + sensitivity),
    }
