"""Train and evaluate the feasibility classifier, the trained model of Module 3.

Run from the backend folder:
    python -m component1.feasibility.train
    python -m component1.feasibility.train --expert component1/data/feasibility_expert.csv

It compares the fixed-rule baseline from the proposal (hand-picked row, GPU and time limits) with
Logistic Regression and Random Forest, on simulated plans labelled by a simulated outcome
(simulate.py). It also runs an ablation without the time and GPU features, and reports how well
the best single feature does, to show the task is not trivially easy. It saves the Random Forest.

READ THIS BEFORE QUOTING ANY NUMBER: the data is simulated. Held-out scores show what the model
learned from the simulation's assumptions. Only --expert (plans labelled by a person who did not
see the model) says anything about real judgement. Report the two separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .simulate import ASSUMPTIONS, FEATURE_NAMES, LABELS, generate_dataset, rule_baseline

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "feasibility_classifier.joblib"
RESULTS_PATH = HERE / "results" / "feasibility_results.json"

WITHOUT_TIME_AND_GPU = [i for i, n in enumerate(FEATURE_NAMES) if n not in ("has_gpu", "weeks_left", "timeline_known")]


def score(y_true, y_pred) -> dict:
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro")), 3),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
    }


def _forest(X, y, seed: int, n_estimators: int):
    search = GridSearchCV(
        RandomForestClassifier(n_estimators=n_estimators, random_state=seed),
        {"min_samples_leaf": [1, 5, 20, 50]}, cv=5, scoring="f1_macro",
    )
    search.fit(X, y)
    return search.best_estimator_, int(search.best_params_["min_samples_leaf"])


def _logistic(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)


def best_single_feature(X, y) -> dict:
    """How far ONE feature gets on its own (a shallow tree). Far below the full model = not trivial."""
    best = (0.0, "")
    for i, name in enumerate(FEATURE_NAMES):
        stump = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[:, [i]], y)
        f1 = float(f1_score(y, stump.predict(X[:, [i]]), labels=LABELS, average="macro"))
        best = max(best, (f1, name))
    return {"feature": best[1], "macro_f1": round(best[0], 3)}


def read_expert(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Plans labelled by a person (see expert_sheet.py). Rows without a label are skipped."""
    from .predict import plan_features_from_values

    X, y = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip().lower()
            if label not in LABELS:
                continue
            x = plan_features_from_values(row["task"], row["algorithm"], row["rows"], row.get("gpu", ""),
                                          row.get("weeks_left", ""))
            if x is not None:
                X.append(x)
                y.append(label)
    return np.array(X), np.array(y)


def train_and_evaluate(n: int = 10000, seed: int = 7, n_estimators: int = 200, save: bool = True,
                       expert: Path | None = None) -> dict:
    X, y = generate_dataset(n, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    forest, leaf = _forest(X_train, y_train, seed, n_estimators)
    logistic = _logistic(X_train, y_train)
    forest_small, _ = _forest(X_train[:, WITHOUT_TIME_AND_GPU], y_train, seed, n_estimators)
    forest_pred = forest.predict(X_test)

    results = {
        "data": {
            "kind": "SIMULATED plans, labelled by a simulated outcome. Not real projects.",
            "n_total": n, "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "class_counts": {l: int((y == l).sum()) for l in LABELS},
            "assumptions": ASSUMPTIONS,
        },
        "best_single_feature": best_single_feature(X_train, y_train),
        "rule_baseline": score(y_test, rule_baseline(X_test)),
        "logistic_regression": score(y_test, logistic.predict(X_test)),
        "random_forest": {**score(y_test, forest_pred), "min_samples_leaf": leaf},
        "ablation_without_time_and_gpu": score(y_test, forest_small.predict(X_test[:, WITHOUT_TIME_AND_GPU])),
        "random_forest_feature_importance": {
            name: round(float(v), 3) for name, v in sorted(zip(FEATURE_NAMES, forest.feature_importances_),
                                                           key=lambda kv: -kv[1])
        },
        "random_forest_confusion_matrix": {
            "labels": LABELS,
            "rows_are_true_columns_are_predicted": confusion_matrix(y_test, forest_pred, labels=LABELS).tolist(),
        },
        "random_forest_per_class_report": classification_report(
            y_test, forest_pred, labels=LABELS, output_dict=True, zero_division=0),
    }
    if expert is not None:
        X_e, y_e = read_expert(expert)
        results["expert_check"] = (
            {"n": int(len(y_e)), "random_forest": score(y_e, forest.predict(X_e)),
             "rule_baseline": score(y_e, rule_baseline(X_e))}
            if len(y_e) else {"n": 0, "note": "no labelled rows in the expert sheet yet"}
        )

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": forest, "labels": LABELS, "features": FEATURE_NAMES}, MODEL_PATH)
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert", type=Path, help="CSV of plans labelled by a person (see expert_sheet.py)")
    args = ap.parse_args(argv)
    r = train_and_evaluate(expert=args.expert)
    print("SIMULATED DATA. These numbers do not prove the model works on real projects.\n")
    print("Class counts:", r["data"]["class_counts"])
    print(f"Best single feature ({r['best_single_feature']['feature']}): macro F1 {r['best_single_feature']['macro_f1']:.3f}")
    for name in ("rule_baseline", "logistic_regression", "random_forest", "ablation_without_time_and_gpu"):
        print(f"  {name:<32} macro F1 {r[name]['macro_f1']:.3f}   accuracy {r[name]['accuracy']:.3f}")
    print("\nFeature importance:", r["random_forest_feature_importance"])
    print("\nConfusion matrix (rows = true, columns = predicted), labels:", LABELS)
    for label, row in zip(LABELS, r["random_forest_confusion_matrix"]["rows_are_true_columns_are_predicted"]):
        print(f"  {label:<11}", row)
    if "expert_check" in r:
        print("\nExpert check:", r["expert_check"])
    print(f"\nSaved model to {MODEL_PATH}\nSaved results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
