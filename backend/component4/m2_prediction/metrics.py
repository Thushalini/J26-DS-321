"""Validation levels 2 and 3: classification and probability-quality metrics."""
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from ..common import config


def expected_calibration_error(y_true, proba, bins: int) -> float:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(proba, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(np.mean(y_true[mask]) - np.mean(proba[mask]))
    return float(total)


def evaluate(y_true, proba) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    predicted = (proba >= config.DECISION_THRESHOLD).astype(int)
    bins = config.CALIBRATION["bins"]
    observed, mean_predicted = calibration_curve(y_true, proba, n_bins=bins, strategy="uniform")

    return {
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, predicted, average="macro", zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "ece": round(expected_calibration_error(y_true, proba, bins), 4),
        "calibration_curve": {
            "mean_predicted": [round(float(v), 4) for v in mean_predicted],
            "observed_rate": [round(float(v), 4) for v in observed],
        },
    }
