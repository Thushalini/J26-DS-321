"""Train and evaluate the context state classifier (feature 4).

Run from the backend folder:
    python -m component1.state.train

It compares a fixed-threshold rule, a Logistic Regression and a Random Forest, each on
two feature sets: the four named in the proposal, and those four plus two extra
signals (an ablation). It saves the best-configured Random Forest and a results file.

READ THIS BEFORE QUOTING ANY NUMBER: the data is synthetic. A score here shows the
pipeline works and what information the state needs. It does NOT show the model is
good on real chat. That needs a real labeled test set the model never trains on.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .simulate import FEATURE_NAMES, LABELS, PROPOSAL_FEATURES, generate_dataset

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "state_classifier.joblib"
RESULTS_PATH = HERE / "results" / "state_classifier_results.json"

FEATURE_SETS = {
    "proposal_4_features": list(range(len(PROPOSAL_FEATURES))),
    "extended_6_features": list(range(len(FEATURE_NAMES))),
}


def rule_baseline(X) -> np.ndarray:
    """A fixed-threshold rule, the kind a team writes by hand. Thresholds are guesses.
    It uses only the first four columns."""
    out = []
    for row in X:
        source_count, agreement, mean_confidence = row[0], row[1], row[2]
        if agreement < 0.7:
            out.append("conflicting")
        elif source_count >= 2 and agreement >= 0.9:
            out.append("confirmed")
        elif mean_confidence < 0.7:
            out.append("uncertain")
        else:
            out.append("known")
    return np.array(out)


def score(y_true, y_pred) -> dict:
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro")), 3),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
    }


def _fit_forest(X_train, y_train, seed: int, n_estimators: int):
    """Pick min_samples_leaf by cross-validation on the TRAINING data only.
    The default (1) lets the forest memorise noise."""
    search = GridSearchCV(
        RandomForestClassifier(n_estimators=n_estimators, random_state=seed),
        {"min_samples_leaf": [1, 5, 20, 50]},
        cv=5,
        scoring="f1_macro",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, int(search.best_params_["min_samples_leaf"])


def train_and_evaluate(n: int = 5000, seed: int = 42, n_estimators: int = 200, save: bool = True) -> dict:
    X, y = generate_dataset(n, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    results = {
        "data": {
            "kind": "SYNTHETIC. Not real chat.",
            "n_total": n,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "label_noise": 0.05,
            "class_counts": {l: int((y == l).sum()) for l in LABELS},
        },
        "rule_baseline": score(y_test, rule_baseline(X_test)),
        "feature_sets": {},
    }

    best_forest = None
    for set_name, cols in FEATURE_SETS.items():
        names = [FEATURE_NAMES[c] for c in cols]
        logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        logistic.fit(X_train[:, cols], y_train)
        forest, leaf = _fit_forest(X_train[:, cols], y_train, seed, n_estimators)
        forest_pred = forest.predict(X_test[:, cols])
        results["feature_sets"][set_name] = {
            "features": names,
            "logistic_regression": score(y_test, logistic.predict(X_test[:, cols])),
            "random_forest": {**score(y_test, forest_pred), "min_samples_leaf": leaf},
            "random_forest_feature_importance": {
                nm: round(float(v), 3) for nm, v in zip(names, forest.feature_importances_)
            },
            "random_forest_confusion_matrix": {
                "labels": LABELS,
                "rows_are_true_columns_are_predicted": confusion_matrix(
                    y_test, forest_pred, labels=LABELS
                ).tolist(),
            },
            "random_forest_per_class_report": classification_report(
                y_test, forest_pred, labels=LABELS, output_dict=True, zero_division=0
            ),
        }
        if set_name == "extended_6_features":
            best_forest = (forest, names)

    if save and best_forest:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": best_forest[0], "labels": LABELS, "features": best_forest[1]}, MODEL_PATH)
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    r = train_and_evaluate()
    print("SYNTHETIC DATA. These numbers do not prove the model works on real chat.\n")
    print("Class counts:", r["data"]["class_counts"])
    print(f"\nFixed rule baseline      macro F1 {r['rule_baseline']['macro_f1']:.3f}   accuracy {r['rule_baseline']['accuracy']:.3f}")
    for set_name, s in r["feature_sets"].items():
        print(f"\n{set_name}: {s['features']}")
        print(f"  logistic regression    macro F1 {s['logistic_regression']['macro_f1']:.3f}   accuracy {s['logistic_regression']['accuracy']:.3f}")
        rf = s["random_forest"]
        print(f"  random forest          macro F1 {rf['macro_f1']:.3f}   accuracy {rf['accuracy']:.3f}   (min_samples_leaf={rf['min_samples_leaf']})")
        print("  forest importance:", s["random_forest_feature_importance"])
    ext = r["feature_sets"]["extended_6_features"]["random_forest_confusion_matrix"]
    print("\nConfusion matrix, extended features (rows = true, columns = predicted):")
    print("labels:", LABELS)
    for label, row in zip(LABELS, ext["rows_are_true_columns_are_predicted"]):
        print(f"  {label:<12}", row)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved results to {RESULTS_PATH}")