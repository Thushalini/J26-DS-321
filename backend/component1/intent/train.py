"""Train and evaluate the decision-intent classifier (feature 8), classical version.

Run from the backend folder:
    python -m component1.intent.train component1/data/intent_synthetic.csv
    python -m component1.intent.train component1/data/intent_synthetic.csv --human component1/data/intent_human.csv

It trains a Logistic Regression on word and character patterns of the message and of the
previous message, and compares it with a majority-class guess and the keyword baseline from
the labeling guide. It also runs the ablation "message alone" against "message plus context".

READ THIS BEFORE QUOTING ANY NUMBER:
  * "held-out" scores come from conversations of the SAME source as the training data. If that
    source is generated text, the score only shows the model learned the generator's style.
  * Only the --human file (messages written and labeled by people) says anything about how the
    model does on messages it was not built from. Report the two results separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from .data import LABELS, Turn, group_split, load_turns
from .features import ContextFeatures
from .keywords import keyword_intent

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "intent_classifier.joblib"
RESULTS_PATH = HERE / "results" / "intent_results.json"


def macro_f1(y_true, y_pred) -> tuple[float, list[str]]:
    """Macro F1 over the labels that really occur in y_true. Returns (score, labels_used)."""
    present = [l for l in LABELS if l in set(y_true)]
    return round(float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)), 3), present


def score(y_true, y_pred) -> dict:
    f1, used = macro_f1(y_true, y_pred)
    return {"macro_f1": f1, "accuracy": round(float(accuracy_score(y_true, y_pred)), 3), "labels_scored": used}


def fit_model(train: list[Turn], use_context: bool, seed: int):
    features = ContextFeatures(use_context=use_context).fit(train)
    X = features.transform(train)
    y = [t.intent for t in train]
    groups = [t.conv_id for t in train]
    folds = max(2, min(5, len(set(groups))))
    search = GridSearchCV(
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        {"C": [0.5, 1, 2, 5, 10, 20]},
        cv=GroupKFold(n_splits=folds),
        scoring="f1_macro",
    )
    search.fit(X, y, groups=groups)
    return features, search.best_estimator_, float(search.best_params_["C"])


def evaluate(turns: list[Turn], features: ContextFeatures, model, training_labels) -> dict:
    y_true = [t.intent for t in turns]
    y_pred = list(model.predict(features.transform(turns)))
    majority = Counter(training_labels).most_common(1)[0][0]
    keyword = [keyword_intent(t.message, t.prev_message) for t in turns]
    return {
        "n_messages": len(turns),
        "class_counts": {l: y_true.count(l) for l in LABELS},
        "majority_class_guess": score(y_true, [majority] * len(turns)),
        "keyword_baseline": score(y_true, keyword),
        "logistic_regression": score(y_true, y_pred),
        "per_class": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": LABELS,
            "rows_are_true_columns_are_predicted": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
    }


def train_and_evaluate(
    train_csv,
    human_csv=None,
    seed: int = 42,
    save: bool = True,
    model_path: Path = MODEL_PATH,
    results_path: Path = RESULTS_PATH,
) -> dict:
    turns = load_turns(train_csv)
    train, heldout = group_split(turns, test_fraction=0.2, seed=seed)

    results: dict = {
        "data": {
            "train_file": str(train_csv),
            "n_train": len(train),
            "n_heldout": len(heldout),
            "n_conversations": len({t.conv_id for t in turns}),
            "train_class_counts": {l: sum(t.intent == l for t in train) for l in LABELS},
            "note": "Held-out scores are from the SAME source as the training data. Only the human test says how it does on people's own messages.",
        },
    }

    chosen = None
    for name, use_context in (("message_only", False), ("with_previous_message", True)):
        features, model, c_value = fit_model(train, use_context, seed)
        report = evaluate(heldout, features, model, [t.intent for t in train])
        report["chosen_C"] = c_value
        results[f"heldout_{name}"] = report
        if use_context:
            chosen = (features, model)

    if human_csv:
        human = load_turns(human_csv)
        results["human_test"] = evaluate(human, chosen[0], chosen[1], [t.intent for t in train])
        results["human_test"]["file"] = str(human_csv)

    if save:
        model_path, results_path = Path(model_path), Path(results_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"features": chosen[0], "model": chosen[1], "labels": LABELS}, model_path)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _line(name: str, s: dict) -> str:
    return f"  {name:<24} macro F1 {s['macro_f1']:.3f}   accuracy {s['accuracy']:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train_csv", help="labeled conversations (for example the generated ones)")
    ap.add_argument("--human", help="messages written and labeled by people: the real test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    r = train_and_evaluate(args.train_csv, args.human, seed=args.seed)
    d = r["data"]
    print(f"{d['n_conversations']} conversations: {d['n_train']} messages to train, {d['n_heldout']} held out.")
    print("Training class counts:", d["train_class_counts"])
    for key, title in (("heldout_message_only", "HELD-OUT, message alone"),
                       ("heldout_with_previous_message", "HELD-OUT, message plus previous message")):
        block = r[key]
        print(f"\n{title} ({block['n_messages']} messages, same source as training):")
        print(_line("majority-class guess", block["majority_class_guess"]))
        print(_line("keyword baseline", block["keyword_baseline"]))
        print(_line("logistic regression", block["logistic_regression"]))
    if "human_test" in r:
        h = r["human_test"]
        print(f"\nHUMAN TEST ({h['n_messages']} messages written and labeled by people):")
        print(_line("majority-class guess", h["majority_class_guess"]))
        print(_line("keyword baseline", h["keyword_baseline"]))
        print(_line("logistic regression", h["logistic_regression"]))
        print("  class counts:", h["class_counts"])
    else:
        print("\nNo human test file given. Add --human to measure on people's own messages.")
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())