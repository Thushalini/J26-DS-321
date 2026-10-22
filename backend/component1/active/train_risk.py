"""Train and evaluate the conflict-risk model on simulated planning histories.

    python -m component1.active.train_risk

Question: will this field be disputed or changed within the next 15 messages? (label from simulate.py)
Split by TEAM, so no team is in both training and test. Compared with:
    rule          what Buddy does now: a value nobody else has agreed with, or one said unsurely, is "at risk"
    logistic_regression, gradient_boosting
The best model on the validation teams (by PR AUC, because only about 1 in 10 rows is positive) is saved
to models/conflict_risk.joblib; all numbers go to results/conflict_risk_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES
from .simulate import ASSUMPTIONS, generate_dataset

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "conflict_risk.joblib"
RESULTS_PATH = HERE / "results" / "conflict_risk_results.json"
N_TEAMS = 600


def rule_score(X: np.ndarray) -> np.ndarray:
    """The baseline: not agreed by anyone else yet, or the last mention was unsure."""
    col = {n: X[:, i] for i, n in enumerate(FEATURE_NAMES)}
    return np.clip((1 - col["is_settled"]) * 0.6 + col["latest_hedged"] * 0.4, 0, 1)


def scores(y, p) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, p)), 3),
            "pr_auc": round(float(average_precision_score(y, p)), 3),
            "brier": round(float(brier_score_loss(y, np.clip(p, 0, 1))), 4)}


def split(groups, seed=0):
    """70% of teams for training, 15% validation, 15% test."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train, rest = next(outer.split(np.zeros(len(groups)), groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val, test = next(inner.split(np.zeros(len(rest)), groups=groups[rest]))
    return train, rest[val], rest[test]


def main(n_teams: int = N_TEAMS, seed: int = 1) -> dict:
    print(f"Simulating {n_teams} teams...")
    X, y, groups = generate_dataset(n_teams, seed=seed)
    tr, va, te = split(groups)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                            l2_regularization=1.0, random_state=0),
    }
    results = {
        "data": {"kind": "SIMULATED planning histories (active/simulate.py). Not real chat.",
                 "teams": n_teams, "rows": int(len(y)), "positive_rate": round(float(y.mean()), 3),
                 "rows_train": int(len(tr)), "rows_validation": int(len(va)), "rows_test": int(len(te)),
                 "assumptions": {k: list(v) if isinstance(v, tuple) else v for k, v in ASSUMPTIONS.items()}},
        "rule": {"validation": scores(y[va], rule_score(X[va])), "test": scores(y[te], rule_score(X[te]))},
    }
    for name, model in models.items():
        model.fit(X[tr], y[tr])
        results[name] = {"validation": scores(y[va], model.predict_proba(X[va])[:, 1]),
                         "test": scores(y[te], model.predict_proba(X[te])[:, 1])}
    best = max(models, key=lambda n: results[n]["validation"]["pr_auc"])
    model = models[best]
    imp = permutation_importance(model, X[te], y[te], scoring="average_precision", n_repeats=5, random_state=0)
    results["best"] = best
    results["feature_importance_test"] = {n: round(float(v), 4) for n, v in
                                          sorted(zip(FEATURE_NAMES, imp.importances_mean), key=lambda kv: -kv[1])}

    # The "safe" reference used to explain a prediction: typical numbers of fields that did NOT change.
    safe = np.median(X[tr][y[tr] == 0], axis=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_NAMES, "safe_reference": safe.tolist(), "name": best}, MODEL_PATH)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("SIMULATED DATA. These numbers do not prove the model works on real teams.\n")
    print(f"Rows: {len(y)}  (positive: {y.mean():.1%})  teams: train {len(set(groups[tr]))}, "
          f"validation {len(set(groups[va]))}, test {len(set(groups[te]))}")
    for name in ("rule", *models):
        t = results[name]["test"]
        print(f"  {name:20s} test ROC AUC {t['roc_auc']:.3f}  PR AUC {t['pr_auc']:.3f}  Brier {t['brier']:.4f}")
    print(f"\nBest on validation: {best}. Most useful features (test): "
          f"{dict(list(results['feature_importance_test'].items())[:6])}")
    print(f"Saved: {MODEL_PATH}")
    return results


if __name__ == "__main__":
    main()
