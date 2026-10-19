"""Trains the baselines and the calibrated stacking ensemble, and scores every model on the test split."""
from sklearn.base import clone

from ..common import config
from . import models
from .data import prediction_features, split
from .metrics import evaluate


def train_and_evaluate(frame, seed: int = config.SEED) -> tuple[dict, object]:
    parts = split(frame, seed)
    x_train, y_train = parts["train"]
    x_val, y_val = parts["val"]
    x_test, y_test = parts["test"]

    results = {"rule_based": evaluate(y_test, models.rule_baseline_proba(x_test))}
    for name, model in models.baselines(seed).items():
        fitted = clone(model).fit(x_train, y_train)
        results[name] = evaluate(y_test, fitted.predict_proba(x_test)[:, 1])

    stack = models.stacking(seed)
    uncalibrated = clone(stack).fit(x_train, y_train)
    results["stacking_uncalibrated"] = evaluate(y_test, uncalibrated.predict_proba(x_test)[:, 1])

    calibrated, method, val_brier = models.calibrate_best(stack, x_train, y_train, x_val, y_val)
    calibrated = models.single_process(calibrated)
    results["stacking_calibrated"] = evaluate(y_test, calibrated.predict_proba(x_test)[:, 1])

    report = {
        "model_version": config.MODEL_VERSION,
        "seed": seed,
        "features": prediction_features(),
        "split_sizes": {name: int(len(y)) for name, (_, y) in parts.items()},
        "test_success_rate": round(float(y_test.mean()), 4),
        "calibration": {"chosen": method, "validation_brier": {k: round(v, 4) for k, v in val_brier.items()}},
        "test_metrics": results,
    }
    return report, calibrated
