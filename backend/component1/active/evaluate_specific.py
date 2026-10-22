"""How well does the specificity check spot values too vague to build on?

    python -m component1.active.evaluate_specific

The values below are INVENTED and labelled by hand (True = too vague), mixing the canonical names
the rules produce with free text an LLM might return. Compared with what Buddy does today: only the
words in rules.VAGUE count as vague. Results go to results/specificity_results.json.
The same person wrote the rules and the labels, so the score is likely optimistic; say so in the report.
First run: accuracy 0.965, recall 0.932 (3 misses, all small words such as "a" and "it" missing from the
generic lists). FUNCTION_WORDS in specific.py was added after that run, so later scores on this set are
measured after a fix made on it.
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..extraction.rules import VAGUE
from .specific import clarify_question

RESULTS = Path(__file__).parent / "results" / "specificity_results.json"

PAIRS = [
    # algorithm
    ("algorithm", "XGBoost", False), ("algorithm", "Random Forest", False), ("algorithm", "LSTM", False),
    ("algorithm", "Logistic Regression", False), ("algorithm", "BERT", False), ("algorithm", "ARIMA", False),
    ("algorithm", "Neural Network", True), ("algorithm", "Transformer", True), ("algorithm", "machine learning", True),
    ("algorithm", "a deep learning model", True), ("algorithm", "some ML model", True),
    ("algorithm", "tree-based model", True), ("algorithm", "ensemble method", True), ("algorithm", "K-Means", False),
    # evaluation metric
    ("evaluation_metric", "RMSE", False), ("evaluation_metric", "F1-score", False), ("evaluation_metric", "MAPE", False),
    ("evaluation_metric", "Accuracy", False), ("evaluation_metric", "performance", True),
    ("evaluation_metric", "how well it does", True), ("evaluation_metric", "error", True),
    ("evaluation_metric", "good results", True), ("evaluation_metric", "ROC AUC", False),
    # task type
    ("task_type", "classification", False), ("task_type", "regression", False), ("task_type", "forecasting", False),
    ("task_type", "prediction", True), ("task_type", "predictive modelling", True), ("task_type", "analysis", True),
    ("task_type", "machine learning", True), ("task_type", "clustering", False),
    # dataset
    ("dataset", "Kaggle hotel bookings dataset", False), ("dataset", "SLTDA monthly arrivals", False),
    ("dataset", "UCI heart disease data", False), ("dataset", "Kaggle data", True), ("dataset", "a Kaggle dataset", True),
    ("dataset", "data from Kaggle", True), ("dataset", "online data", True), ("dataset", "public data", True),
    ("dataset", "government open data", True), ("dataset", "census 2020", False), ("dataset", "the MIMIC-III records", False),
    # target variable
    ("target_variable", "tourist arrivals", False), ("target_variable", "customer churn", False),
    ("target_variable", "house price", False), ("target_variable", "the output", True), ("target_variable", "the label", True),
    ("target_variable", "results", True), ("target_variable", "the prediction", True), ("target_variable", "fraud label", False),
    ("target_variable", "final score", True), ("target_variable", "daily sales", False),
    # domain
    ("domain", "tourism", False), ("domain", "healthcare", False), ("domain", "data science", True),
    ("domain", "business", True), ("domain", "technology", True), ("domain", "agriculture", False),
    ("domain", "machine learning", True),
    # problem / objective
    ("problem", "predicting tourist arrivals for Sri Lanka", False), ("problem", "prediction", True),
    ("problem", "data analysis", True), ("problem", "build a model", True), ("problem", "detecting fraud in card payments", False),
    ("objective", "help hotels plan staff for busy months", False), ("objective", "improve accuracy", True),
    ("objective", "get good results", True), ("objective", "reduce customer churn by 10%", False),
    ("objective", "make better insights", True),
    # resources
    ("resources", "Google Colab", False), ("resources", "GPU", False), ("resources", "CPU only", False),
    ("resources", "the cloud", True), ("resources", "a powerful computer", True), ("resources", "a server", True),
    ("resources", "AWS", False),
    # timeline
    ("timeline", "deadline 2026-12-15", False), ("timeline", "deadline in 2 weeks", False), ("timeline", "4 weeks", False),
    ("timeline", "week 10", False), ("timeline", "soon", True), ("timeline", "end of semester", True),
    ("timeline", "later this year", True), ("timeline", "deadline next month", False),
    # words that are already thrown away today
    ("dataset", "the data", True), ("target_variable", "something", True),
]

# Written and labelled AFTER the fix above, and never used to change the rules: the fairer number.
HOLDOUT = [
    ("algorithm", "CatBoost", False), ("algorithm", "an unsupervised approach", True), ("algorithm", "GRU", False),
    ("algorithm", "the best classifier", True), ("evaluation_metric", "R-squared", False),
    ("evaluation_metric", "overall quality", True), ("task_type", "anomaly detection", False),
    ("task_type", "data science", True), ("dataset", "the Hugging Face IMDB reviews", False),
    ("dataset", "a dataset from GitHub", True), ("dataset", "Kaggle Titanic data", False),
    ("target_variable", "the class", True), ("target_variable", "length of hospital stay", False),
    ("target_variable", "values", True), ("domain", "energy", False), ("domain", "various industries", True),
    ("problem", "forecasting electricity demand in Colombo", False), ("problem", "do some analysis", True),
    ("objective", "cut readmissions for heart patients", False), ("objective", "build a better model", True),
    ("resources", "Kaggle notebooks", False), ("resources", "our machines", True),
    ("timeline", "by the end of the year", True), ("timeline", "deadline 30 November", False),
]


def _today(field, value):
    return value.strip().lower() in VAGUE


def _new(field, value):
    return clarify_question(field, value) is not None


def _score(method, pairs=PAIRS):
    truth = [p[2] for p in pairs]
    guess = [method(f, v) for f, v, _ in pairs]
    wrong = [f"{f}: {v!r} (said {'vague' if g else 'specific'})" for (f, v, t), g in zip(pairs, guess) if g != t]
    return {"accuracy": round(accuracy_score(truth, guess), 3),
            "precision": round(precision_score(truth, guess, zero_division=0), 3),
            "recall": round(recall_score(truth, guess, zero_division=0), 3),
            "f1": round(f1_score(truth, guess, zero_division=0), 3), "mistakes": wrong}


def main():
    results = {"pairs": len(PAIRS), "vague": sum(p[2] for p in PAIRS),
               "today_vague_word_list": _score(_today), "specificity_check": _score(_new),
               "holdout": {"pairs": len(HOLDOUT), "vague": sum(p[2] for p in HOLDOUT),
                           "today_vague_word_list": _score(_today, HOLDOUT),
                           "specificity_check": _score(_new, HOLDOUT)}}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{len(PAIRS)} invented values, {results['vague']} labelled vague.\n")
    for name in ("today_vague_word_list", "specificity_check"):
        r = results[name]
        print(f"  {name:22s} accuracy {r['accuracy']:.3f}  precision {r['precision']:.3f}  "
              f"recall {r['recall']:.3f}  F1 {r['f1']:.3f}")
    print("\nSpecificity check mistakes:")
    for m in results["specificity_check"]["mistakes"] or ["(none)"]:
        print("  ", m)
    h = results["holdout"]
    print(f"\nHoldout: {h['pairs']} values written after the rules were final, {h['vague']} vague.")
    for name in ("today_vague_word_list", "specificity_check"):
        r = h[name]
        print(f"  {name:22s} accuracy {r['accuracy']:.3f}  precision {r['precision']:.3f}  "
              f"recall {r['recall']:.3f}  F1 {r['f1']:.3f}")
    for m in h["specificity_check"]["mistakes"]:
        print("   mistake:", m)


if __name__ == "__main__":
    main()
