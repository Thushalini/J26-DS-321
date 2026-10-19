"""SHAP explanations of the calibrated success probability.

The ensemble is explained as a black box (permutation SHAP over predict_proba), so every
contribution is in probability points and they add up to the gap between the project's
probability and the average project's.
"""
import numpy as np
import pandas as pd
import shap

from ..common import config


def _success_proba(model, columns):
    def predict(values):
        return model.predict_proba(pd.DataFrame(values, columns=columns))[:, 1]
    return predict


def explainer(model, background: pd.DataFrame, seed: int = config.SEED):
    sample = background.sample(min(len(background), config.EXPLAINABILITY["background_size"]), random_state=seed)
    masker = shap.maskers.Independent(sample, max_samples=len(sample))
    return shap.explainers.Permutation(_success_proba(model, list(background.columns)), masker, seed=seed)


def shap_values(model, background: pd.DataFrame, rows: pd.DataFrame, seed: int = config.SEED) -> shap.Explanation:
    explanation = explainer(model, background, seed)(rows, silent=True)
    explanation.feature_names = list(rows.columns)
    return explanation


def global_importance(explanation: shap.Explanation) -> list[dict]:
    scale = config.EXPLAINABILITY["probability_to_points"]
    mean_abs = np.abs(explanation.values).mean(axis=0) * scale
    order = np.argsort(mean_abs)[::-1]
    return [{"feature": explanation.feature_names[i], "mean_abs_points": round(float(mean_abs[i]), 3)} for i in order]


def local_explanation(explanation: shap.Explanation, index: int) -> dict:
    scale = config.EXPLAINABILITY["probability_to_points"]
    values = explanation.values[index]
    order = np.argsort(np.abs(values))[::-1]
    return {
        "base_probability": round(float(explanation.base_values[index]), 4),
        "probability": round(float(explanation.base_values[index] + values.sum()), 4),
        "factors": [
            {
                "feature": explanation.feature_names[i],
                "value": float(explanation.data[index][i]),
                "contribution_points": round(float(values[i] * scale), 2),
            }
            for i in order
        ],
    }
