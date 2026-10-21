"""Hyperparameter search and cross-validated results, both on the training split only (the test split stays unseen)."""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from ..common import config
from . import models


def _folds(n_splits: int, seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def tune_baselines(x_train, y_train, seed: int = config.SEED) -> dict:
    settings = config.TUNING
    tuned, report = {}, {}
    for name, model in models.baselines(seed).items():
        search = GridSearchCV(model, settings["grids"][name], scoring=settings["scoring"],
                              cv=_folds(settings["cv_folds"], seed), n_jobs=-1)
        search.fit(x_train, y_train)
        tuned[name] = search.best_params_
        report[name] = {"best_params": search.best_params_, "cv_score": round(float(search.best_score_), 4)}
    return tuned, report


def _summary(scores: dict) -> dict:
    summary = {}
    for metric in config.CROSS_VALIDATION["scoring"]:
        values = scores[f"test_{metric}"]
        values = -values if metric == "brier" else values
        summary[metric] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}
    return summary


def cross_validated(x_train, y_train, tuned: dict, seed: int = config.SEED) -> dict:
    settings = config.CROSS_VALIDATION
    folds = _folds(settings["folds"], seed)
    candidates = {**models.baselines(seed, tuned), "stacking": models.stacking(seed, tuned)}
    return {
        name: _summary(cross_validate(clone(model), x_train, y_train, cv=folds, scoring=settings["scoring"]))
        for name, model in candidates.items()
    }
