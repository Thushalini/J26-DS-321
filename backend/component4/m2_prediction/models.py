"""Baseline models, the stacking ensemble, the rule-based baseline and calibration."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..common import config


def _logistic_regression(seed: int):
    return make_pipeline(StandardScaler(), LogisticRegression(**config.MODEL_PARAMS["logistic_regression"], random_state=seed))


def baselines(seed: int = config.SEED, tuned: dict | None = None) -> dict:
    params = config.MODEL_PARAMS
    models = {
        "logistic_regression": _logistic_regression(seed),
        "random_forest": RandomForestClassifier(**params["random_forest"], random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(**params["gradient_boosting"], random_state=seed),
    }
    for name, overrides in (tuned or {}).items():
        models[name].set_params(**overrides)
    return models


def stacking(seed: int = config.SEED, tuned: dict | None = None) -> StackingClassifier:
    return StackingClassifier(
        estimators=list(baselines(seed, tuned).items()),
        final_estimator=LogisticRegression(**config.MODEL_PARAMS["logistic_regression"], random_state=seed),
        cv=config.MODEL_PARAMS["stacking_cv"],
        stack_method="predict_proba",
        n_jobs=-1,
    )


def calibrate_best(model, x_train, y_train, x_val, y_val) -> tuple:
    """Fits one calibrated copy per method and keeps the one with the lower validation Brier score."""
    scores, fitted = {}, {}
    for method in config.CALIBRATION["methods"]:
        candidate = CalibratedClassifierCV(clone(model), method=method, cv=config.CALIBRATION["cv"], ensemble=False)
        candidate.fit(x_train, y_train)
        scores[method] = float(brier_score_loss(y_val, candidate.predict_proba(x_val)[:, 1]))
        fitted[method] = candidate
    best = min(scores, key=scores.get)
    return fitted[best], best, scores


def single_process(model):
    """Turns off parallel workers after training: starting them costs more than predicting a few rows."""
    pending = [model]
    while pending:
        current = pending.pop()
        if hasattr(current, "n_jobs"):
            current.n_jobs = None
        for attribute in ("calibrated_classifiers_", "estimators_"):
            children = getattr(current, attribute, None)
            if isinstance(children, list):
                pending.extend(children)
        for attribute in ("estimator", "final_estimator_"):
            if getattr(current, attribute, None) is not None:
                pending.append(getattr(current, attribute))
    return model


def rule_baseline_proba(x: pd.DataFrame) -> np.ndarray:
    """1 = predicted success. A project is "at risk" when all three warning signs are present."""
    rule = config.RULE_BASELINE
    at_risk = (
        (x["drift_score"] >= rule["drift_score_at_least"])
        & (x["issue_count"] >= rule["issue_count_at_least"])
        & (x["reproducibility"] <= rule["reproducibility_at_most"])
    )
    return (~at_risk).astype(float).to_numpy()
