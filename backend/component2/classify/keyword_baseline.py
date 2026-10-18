"""
Step 2b: Keyword baseline classifier.

The simplest possible way to guess a lifecycle stage: look for typical
words, code and file names for each stage, count the hits, pick the stage
with the most. Our real model (Step 2c) must beat this to prove it adds value.

Run from backend:
    python -m component2.classify.keyword_baseline
"""
import re
from collections import Counter

from component2.db import connect, save_labels

# Each stage -> clues. Plain words match whole words; entries with ( or . match as written.
KEYWORDS = {
    "data_cleaning": [
        "clean", "cleaning", "missing", "null", "nan", "dropna", "fillna", "drop_duplicates",
        "duplicate", "outlier", "dtype", "astype", "read_csv", "read_excel", "merge", "preprocess",
        "preprocessing", "impute", "dataset", "scrape", "load data", "raw data", "to_datetime",
    ],
    "eda": [
        "eda", "explore", "exploratory", "describe(", "info(", "value_counts", "histogram", "hist",
        "histplot", "boxplot", "scatter", "heatmap", "corr(", "correlation", "distribution",
        "visualize", "visualization", "plot", "plt.", "sns.", "countplot", "pairplot", "analysis",
    ],
    "feature_engineering": [
        "feature", "features", "encode", "encoding", "get_dummies", "onehot", "one-hot", "labelencoder",
        "standardscaler", "minmaxscaler", "scaler", "normalize", "normalise", "train_test_split",
        "pca", "selectkbest", "feature selection", "tfidf", "vectorizer", "tokenize", "embedding",
    ],
    "model_development": [
        "model", "train", "training", ".fit(", "fit(", "classifier", "regressor", "randomforest",
        "random forest", "xgboost", "xgbclassifier", "lightgbm", "logisticregression", "svm", "knn",
        "neural", "keras", "tensorflow", "torch", "epoch", "epochs", "hyperparameter", "gridsearchcv",
        "tune", "tuning", "baseline", "lstm", "cnn", "bert", "yolo",
    ],
    "model_evaluation": [
        "evaluate", "evaluation", "accuracy", "precision", "recall", "f1", "f1_score", "auc", "roc",
        "confusion_matrix", "classification_report", "cross_val_score", "cross-validation",
        "validation", "metric", "metrics", "mse", "rmse", "mae", "r2_score", "score(", "overfit",
        "test set", "predict(",
    ],
    "deployment_mlops": [
        "deploy", "deployment", "docker", "dockerfile", "api", "fastapi", "flask", "streamlit",
        "endpoint", "pipeline", "ci", "workflow", ".yml", "requirements.txt", "environment.yml",
        "pickle", "joblib", "mlflow", "serve", "server", "app.py", "heroku", "gradio",
    ],
    "documentation": [
        "readme", "readme.md", "docs", "documentation", "report", "comment", "comments", "docstring",
        "typo", "slides", "presentation", "write-up", "writeup", "license", ".md", "notes",
    ],
}


def _make_pattern(clue):
    if re.search(r"[(.\-_/]", clue):           # code-like clue: match literally
        return re.compile(re.escape(clue), re.I)
    return re.compile(r"\b" + re.escape(clue) + r"\b", re.I)


PATTERNS = {stage: [_make_pattern(c) for c in clues] for stage, clues in KEYWORDS.items()}


def classify(text, cell_type=None):
    """Returns (stage, confidence). confidence = share of all keyword hits won by the top stage."""
    if cell_type == "markdown" and len(text) < 200:
        return "documentation", 0.6
    hits = Counter({stage: sum(1 for p in pats if p.search(text)) for stage, pats in PATTERNS.items()})
    total = sum(hits.values())
    if total == 0:
        return "other", 0.0
    stage, best = hits.most_common(1)[0]
    return stage, round(best / total, 3)


def main():
    import json
    conn = connect()
    rows = conn.execute("SELECT activity_id, text, meta FROM activities").fetchall()
    labels = []
    for activity_id, text, meta in rows:
        cell_type = json.loads(meta or "{}").get("cell_type")
        stage, conf = classify(text, cell_type)
        labels.append((activity_id, stage, conf))
    save_labels(conn, "baseline", labels)

    counts = Counter(stage for _, stage, _ in labels)
    print(f"Labelled {len(labels)} activities with the keyword baseline:")
    for stage, n in counts.most_common():
        print(f"  {stage:<22} {n:>5}  ({n / len(labels):.0%})")


if __name__ == "__main__":
    main()