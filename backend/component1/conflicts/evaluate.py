"""How well does each method tell "same choice" from "a real disagreement"?

    python -m component1.conflicts.evaluate

The pairs below are INVENTED for this check (no real chat), labelled by hand: True = same choice.
Three methods are compared:
    exact      the same spelling after normalising (what Component 1 did before)
    names      exact + known names and abbreviations (RF = random forest)
    names+nli  names + the NLI rule for free-text fields (conflicts/same.py)
Results go to component1/conflicts/results/conflict_results.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from ..state.normalize import normalize_value
from . import same as same_module

RESULTS = Path(__file__).parent / "results" / "conflict_results.json"

PAIRS = [
    # field, a, b, same?
    ("algorithm", "RF", "random forest", True),
    ("algorithm", "GBM", "gradient boosting", True),
    ("algorithm", "ANN", "neural network", True),
    ("algorithm", "svc", "SVM", True),
    ("algorithm", "random forest", "XGBoost", False),
    ("algorithm", "logistic regression", "decision tree", False),
    ("evaluation_metric", "f-measure", "F1", True),
    ("evaluation_metric", "F1", "accuracy", False),
    ("evaluation_metric", "RMSE", "MAE", False),
    ("task_type", "classification", "classifying", True),
    ("task_type", "regression", "classification", False),
    ("target_variable", "tourist arrivals", "hotel occupancy", False),
    ("target_variable", "monthly tourist arrivals", "tourist arrivals", True),
    ("target_variable", "customer churn", "churn", True),
    ("target_variable", "house price", "sale price of the house", True),
    ("target_variable", "house price", "number of bedrooms", False),
    ("target_variable", "loan default", "credit score", False),
    ("target_variable", "daily sales", "sales per day", True),
    ("dataset", "Kaggle hotel bookings", "hotel bookings data from Kaggle", True),
    ("dataset", "Kaggle hotel bookings", "census 2020", False),
    ("dataset", "UCI heart disease", "heart disease data from UCI", True),
    ("dataset", "twitter posts", "hospital records", False),
    ("problem", "predicting tourist arrivals", "forecasting how many tourists will arrive", True),
    ("problem", "predicting tourist arrivals", "detecting fake reviews", False),
    ("problem", "spotting fraudulent transactions", "fraud detection in payments", True),
    ("problem", "spotting fraudulent transactions", "recommending movies", False),
    ("domain", "tourism", "travel and tourism", True),
    ("domain", "tourism", "healthcare", False),
    ("domain", "finance", "banking and finance", True),
    ("objective", "reduce customer churn", "lower the number of customers who leave", True),
    ("objective", "reduce customer churn", "increase hotel bookings", False),
    ("objective", "improve hotel planning", "help hotels plan staff", True),
]


def _exact(field, a, b):
    return normalize_value(a) == normalize_value(b)


def _names(field, a, b):
    if _exact(field, a, b):
        return True
    na, nb = same_module.known_names(field, a), same_module.known_names(field, b)
    return bool(na and nb and na[0] == nb[0])


def _score(method, pairs):
    truth = [p[3] for p in pairs]
    guess = [method(f, a, b) for f, a, b, _ in pairs]
    wrong = [f"{f}: {a!r} vs {b!r} (said {'same' if g else 'different'})"
             for (f, a, b, t), g in zip(pairs, guess) if g != t]
    return {"accuracy": round(accuracy_score(truth, guess), 3),
            "macro_f1": round(f1_score(truth, guess, average="macro"), 3), "mistakes": wrong}


def main():
    os.environ["C1_NLI"] = "on"
    free = [p for p in PAIRS if p[0] in same_module.FREE_TEXT]
    results = {"pairs": len(PAIRS), "free_text_pairs": len(free), "methods": {}}
    for name, method in (("exact", _exact), ("names", _names), ("names+nli", same_module.same_value)):
        results["methods"][name] = {"all": _score(method, PAIRS), "free_text": _score(method, free)}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, r in results["methods"].items():
        print(f"{name:10s} all: acc {r['all']['accuracy']:.3f} F1 {r['all']['macro_f1']:.3f} | "
              f"free text: acc {r['free_text']['accuracy']:.3f} F1 {r['free_text']['macro_f1']:.3f}")
    print("\nnames+nli mistakes:")
    for m in results["methods"]["names+nli"]["all"]["mistakes"]:
        print("  ", m)


if __name__ == "__main__":
    main()
