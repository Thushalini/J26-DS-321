"""
Step 3c: Planted-drift test set (objective O3).

We build many small test "windows" where WE decide the truth:
  - a planned value (written the way a Component 1 blueprint might write it)
  - 10 activity texts, of which a chosen share talk about something else
    (e.g. 3 of 10 algorithm mentions are XGBoost instead of Random Forest)
Because we planted the change, we know the correct drift score and severity.

Then both methods judge every window:
  - OUR method  : detect_drift.score_dimension (short names + MiniLM meaning)
  - BASELINE    : exact keyword - a mention is on plan only if the blueprint
                  words appear literally in the text
and we report accuracy for each. Target: >= 0.80 detection accuracy.
The same windows are judged again written as code diffs (each text among ordinary code lines),
because drift is also read from the code lines commits add to .py files.

Run from backend:
    python -m component2.drift.evaluate_drift
"""
import csv
import random
from collections import defaultdict

import component2.drift.detect_drift as detector
from component2.config import DATA_DIR
from component2.drift.detect_drift import score_dimension, severity_of
from component2.drift.mentions import count_mentions, find_mentions

SEED = 7
TEXTS_PER_WINDOW = 10
OFF_PLAN_SHARES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]   # truth: none, none, minor, moderate, significant, significant
RESULTS_DIR = DATA_DIR / "results"

# For each dimension: canonical name -> (ways a blueprint writes it, ways a team writes it in commits/code)
CASES = {
    "algorithm": {
        "random forest": (["Random Forest", "Random Forest Classifier", "RF"],
                          ["model = RandomForestClassifier(n_estimators=200)", "Train RF baseline",
                           "tuned random forest max_depth"]),
        "xgboost": (["XGBoost", "XGBoost classifier"],
                    ["clf = XGBClassifier(eta=0.1)", "switch to xgboost", "XGB early stopping added"]),
        "logistic regression": (["Logistic Regression", "logistic regression model"],
                                ["lr = LogisticRegression(max_iter=500)", "logistic regression baseline done"]),
        "svm": (["Support Vector Machine", "SVM"], ["svc = SVC(kernel='rbf')", "try SVM with rbf kernel"]),
        "knn": (["K-Nearest Neighbours", "KNN"], ["KNeighborsClassifier(n_neighbors=5)", "knn with k=7"]),
        "decision tree": (["Decision Tree"], ["DecisionTreeClassifier(max_depth=4)", "pruned the decision tree"]),
        "linear regression": (["Linear Regression"], ["reg = LinearRegression().fit(X, y)", "linear regression residuals"]),
        "neural network": (["Neural Network", "MLP neural network"],
                           ["MLPClassifier(hidden_layer_sizes=(64,))", "neural net with 2 hidden layers"]),
    },
    "metric": {
        "f1-score": (["F1-score", "F1 score (macro)"],
                     ["print(f1_score(y_test, pred))", "F1 is 0.81 on test", "report macro f1"]),
        "accuracy": (["Accuracy"], ["accuracy_score(y_test, pred)", "accuracy is 0.91", "check accuracy again"]),
        "roc auc": (["ROC-AUC", "AUC"], ["roc_auc_score(y_test, proba)", "AUC = 0.88", "plot the ROC curve"]),
        "rmse": (["RMSE", "Root Mean Squared Error"], ["rmse on validation is 3.2", "root mean squared error improved"]),
        "mae": (["MAE", "Mean Absolute Error"], ["mean_absolute_error(y, p)", "MAE dropped to 2.1"]),
        "recall": (["Recall"], ["recall_score(y_test, pred)", "recall for churners is low"]),
    },
    "target": {
        "churn": (["Churn"], ["y = df['Churn']", "target = 'Churn'", "target variable is Churn"]),
        "saleprice": (["SalePrice", "Sale Price"], ["y = df['SalePrice']", "target = 'SalePrice'"]),
        "survived": (["Survived"], ["y = data['Survived']", "label = 'Survived'"]),
        "default": (["default"], ["y = df['default']", "target_col = 'default'"]),
        "tenure": (["tenure"], ["y = df['tenure']", "target = 'tenure'"]),
    },
}


# Ordinary code lines with no algorithm / metric / target in them, to bury each text inside a "code diff"
BOILERPLATE = [
    "import pandas as pd", "import numpy as np", "import matplotlib.pyplot as plt",
    "df = pd.read_csv('data/train.csv')", "df = df.dropna()", "print(df.shape)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
    "plt.figure(figsize=(8, 5))", "plt.savefig('out.png')", "def main():", "    return result",
]


def as_code_diffs(windows, rng):
    """Same windows, but every text becomes a few added code lines with the mention among them,
    like the code_change activities read from a commit's .py files."""
    out = []
    for w in windows:
        texts = []
        for t in w["texts"]:
            lines = rng.sample(BOILERPLATE, 4) + [t]
            rng.shuffle(lines)
            texts.append("\n".join(lines))
        out.append({**w, "texts": texts})
    return out


def baseline_score(planned, texts, dim):
    """Exact keyword: of the texts about this dimension, how many do NOT contain the plan word-for-word?"""
    about = [t for t in texts if find_mentions(t)[dim]]
    if not about:
        return None
    off = sum(1 for t in about if planned.lower() not in t.lower())
    return off / len(about)


def build_windows(rng):
    windows = []
    for dim, options in CASES.items():
        names = list(options)
        for name, (blueprint_ways, team_ways) in options.items():
            others = [n for n in names if n != name]
            for planned in blueprint_ways:
                for share in OFF_PLAN_SHARES:
                    n_off = round(share * TEXTS_PER_WINDOW)
                    new_name = rng.choice(others)
                    texts = [rng.choice(team_ways) for _ in range(TEXTS_PER_WINDOW - n_off)]
                    texts += [rng.choice(options[new_name][1]) for _ in range(n_off)]
                    rng.shuffle(texts)
                    windows.append({"dimension": dim, "planned": planned, "changed_to": new_name if n_off else "",
                                    "true_score": n_off / TEXTS_PER_WINDOW, "texts": texts})
    return windows


def judge(windows):
    rows = []
    for w in windows:
        truth = severity_of(w["true_score"])
        seen = count_mentions(w["texts"])[w["dimension"]]
        ours, our_sev, _, why = score_dimension(w["dimension"], w["planned"], seen)
        base = baseline_score(w["planned"], w["texts"], w["dimension"])
        base_sev = severity_of(base) if base is not None else "no_evidence"
        rows.append({"dimension": w["dimension"], "planned": w["planned"], "changed_to": w["changed_to"],
                     "true_score": w["true_score"], "true_severity": truth,
                     "our_score": ours, "our_severity": our_sev,
                     "baseline_score": None if base is None else round(base, 3), "baseline_severity": base_sev,
                     "explanation": why})
    return rows


def accuracy(rows, method, level):
    if level == "detect":       # drift yes/no
        ok = [(r[f"{method}_severity"] != "none") == (r["true_severity"] != "none") for r in rows]
    else:                       # exact severity level
        ok = [r[f"{method}_severity"] == r["true_severity"] for r in rows]
    return sum(ok) / len(ok)


def main():
    windows = build_windows(random.Random(SEED))

    # 1) Choose the similarity threshold from data instead of guessing
    print("Similarity threshold (MATCH_AT) vs detection accuracy:")
    for m in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        detector.MATCH_AT = m
        print(f"  MATCH_AT = {m:.2f}  ->  {accuracy(judge(windows), 'our', 'detect'):.3f}")
    detector.MATCH_AT = 0.75
    print()

    # 2) Full results at the chosen threshold
    rows = judge(windows)
    groups = defaultdict(list)
    for r in rows:
        groups[r["dimension"]].append(r)
    groups["ALL"] = rows

    print(f"Planted-drift test set: {len(rows)} windows x {TEXTS_PER_WINDOW} texts\n")
    print(f"{'dimension':<10} {'windows':>7} | {'detect: ours':>12} {'baseline':>9} | {'severity: ours':>14} {'baseline':>9}")
    print("-" * 72)
    for dim in ["algorithm", "metric", "target", "ALL"]:
        g = groups[dim]
        print(f"{dim:<10} {len(g):>7} | {accuracy(g, 'our', 'detect'):>12.2f} {accuracy(g, 'baseline', 'detect'):>9.2f} | "
              f"{accuracy(g, 'our', 'severity'):>14.2f} {accuracy(g, 'baseline', 'severity'):>9.2f}")

    false_alarms_base = sum(1 for r in rows if r["true_severity"] == "none" and r["baseline_severity"] != "none")
    false_alarms_ours = sum(1 for r in rows if r["true_severity"] == "none" and r["our_severity"] != "none")
    print(f"\nFalse alarms on windows with NO real drift: ours {false_alarms_ours}, baseline {false_alarms_base}")

    # 3) The same windows written as code diffs (mentions buried among ordinary code lines)
    diff_rows = judge(as_code_diffs(windows, random.Random(SEED)))
    print(f"\nSame windows as code diffs ({len(BOILERPLATE)} kinds of boilerplate line mixed in):")
    print(f"  detect: ours {accuracy(diff_rows, 'our', 'detect'):.2f}  baseline {accuracy(diff_rows, 'baseline', 'detect'):.2f}"
          f"  | severity: ours {accuracy(diff_rows, 'our', 'severity'):.2f}  baseline {accuracy(diff_rows, 'baseline', 'severity'):.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "drift_evaluation.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Every window's result saved to {out}")


if __name__ == "__main__":
    main()