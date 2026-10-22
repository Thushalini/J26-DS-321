"""
Step 3a: find what a team is actually talking about / coding for the three
blueprint decisions: the TARGET variable, the ALGORITHM, and the METRIC.

Many ways of writing the same thing are mapped to one name, e.g.
"RF", "RandomForestClassifier(", "random forest"  ->  "random forest".
"""
import re
from collections import Counter

ALGORITHMS = {
    "random forest": [r"random\s*forest", r"RandomForest\w*", r"\bRF\b"],
    "xgboost": [r"xgboost", r"\bXGB\w*"],
    "lightgbm": [r"lightgbm", r"\bLGBM\w*"],
    "catboost": [r"catboost"],
    "gradient boosting": [r"gradient\s*boost\w*", r"GradientBoosting\w*"],
    "decision tree": [r"decision\s*tree", r"DecisionTree\w*"],
    "logistic regression": [r"logistic\s*regression", r"LogisticRegression"],
    "linear regression": [r"linear\s*regression", r"LinearRegression", r"\bRidge\b", r"\bLasso\b"],
    "svm": [r"\bSVM\b", r"\bSVC\b", r"\bSVR\b", r"support\s*vector"],
    "knn": [r"\bKNN\b", r"k-?nearest", r"KNeighbors\w*"],
    "naive bayes": [r"naive\s*bayes", r"\w+NB\("],
    "neural network": [r"neural\s*net\w*", r"\bMLP\w*", r"\bkeras\b", r"Sequential\("],
    "cnn": [r"\bCNN\b", r"Conv2D", r"convolutional"],
    "lstm": [r"\bLSTM\b"],
    "bert": [r"\bBERT\b", r"roberta", r"distilbert"],
    "kmeans": [r"k-?means", r"KMeans"],
    "isolation forest": [r"isolation\s*forest", r"IsolationForest"],
    "stacking": [r"stacking", r"Stacking(?:Classifier|Regressor)"],
    "yolo": [r"\bYOLO\w*"],
}

METRICS = {
    "accuracy": [r"\baccuracy", r"accuracy_score"],
    "f1-score": [r"\bf1\b", r"f1[_\- ]?score", r"\bf1_(?:macro|micro|weighted)"],
    "precision": [r"(?<!average[\s_])\bprecision"],     # not "(mean) average precision" = mAP
    "recall": [r"\brecall(?:_score)?\b"],
    "roc auc": [r"\bauc\b", r"\broc\b", r"roc_auc"],
    "rmse": [r"\brmse\b", r"root\s*mean\s*squared"],
    "mse": [r"\bmse\b", r"mean_squared_error"],
    "mae": [r"\bmae\b", r"mean_absolute_error", r"mean\s*absolute\s*error"],
    "r2": [r"\br2\b", r"r2_score", r"r-squared"],
    "log loss": [r"log\s*loss", r"log_loss"],
    # (?-i:...) = case-sensitive, so pandas .map( and the word "map" are not counted as the metric
    "map": [r"(?-i:\bmAP(?![a-z]))", r"average[\s_]precision"],    # also mAP50, mAP@0.5
    "iou": [r"\bIoU\b"],
}

# Target column: only clear code / text patterns, to avoid guessing
TARGET_PATTERNS = [
    r"\by\s*=\s*\w+\s*\[\s*['\"]([\w ]+)['\"]\s*\]",                 # y = df['Churn']
    r"\b(?:target|label|target_col|label_col)\w*\s*=\s*['\"]([\w ]+)['\"]",  # target = 'Churn'
    r"target\s+(?:variable|column)?\s*(?:is|:)\s*['\"]?([A-Za-z_]\w+)",   # target variable is Churn
]

_compiled = {
    "algorithm": {name: [re.compile(p, re.I) for p in pats] for name, pats in ALGORITHMS.items()},
    "metric": {name: [re.compile(p, re.I) for p in pats] for name, pats in METRICS.items()},
}
_target = [re.compile(p, re.I) for p in TARGET_PATTERNS]

def normalise_target(name):
    """'Sale Price', 'sale_price', 'SalePrice' -> 'saleprice' so they compare equal."""
    return re.sub(r"[\s_\-]", "", name.strip().lower())

def find_mentions(text):
    """Returns {'target': Counter, 'algorithm': Counter, 'metric': Counter} for one text."""
    found = {"target": Counter(), "algorithm": Counter(), "metric": Counter()}
    for dim in ("algorithm", "metric"):
        for name, pats in _compiled[dim].items():
            if any(p.search(text) for p in pats):
                found[dim][name] += 1
    for p in _target:
        for m in p.findall(text):
            found["target"][normalise_target(m)] += 1
    return found


def count_mentions(texts):
    total = {"target": Counter(), "algorithm": Counter(), "metric": Counter()}
    for t in texts:
        for dim, c in find_mentions(t).items():
            total[dim].update(c)
    return total