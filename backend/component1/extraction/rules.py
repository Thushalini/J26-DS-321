"""Rule-based extraction: word lists and phrase patterns, no language model.

Runs on this machine in milliseconds, costs nothing, sends no chat anywhere, and
gives the same answer every time. Every result can be traced to the rule that made it.
It is also the baseline the LLM extractor is compared against in the report.

What it covers well: algorithm, evaluation_metric, task_type (known names), and hedging.
Fairly well: target_variable, dataset, domain, resources (phrase patterns).
Weakly: problem and objective (only when stated as "the problem is ..." / "the goal is ...").
Free wording it does not recognise is left for the LLM fallback (see hybrid.py).

Known limits (say so in the report):
  - Synonyms and abbreviations outside the lists are missed ("RF", "GBM").
  - A rejected option ("not XGBoost") is skipped, but rejection is not recorded.
  - If one message names two values for the same field, only the first is kept and it is
    marked hedged, because the message is probably comparing options, not deciding.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from ..common.schema import FieldName
from ..timeline.find import find_timeline
from .extractor import ExtractedItem
from .size import find_sizes

# ------------------------------------------------------------------ word lists
# canonical value -> spellings to look for (matched as whole words, any case)
ALGORITHMS: dict[str, list[str]] = {
    "XGBoost": ["xgboost", "xg boost", "xgb"],
    "LightGBM": ["lightgbm", "light gbm"],
    "CatBoost": ["catboost"],
    "Random Forest": ["random forests?", "random-forests?"],
    "Decision Tree": ["decision trees?"],
    "Gradient Boosting": ["gradient boosting", "gradient boosted trees"],
    "AdaBoost": ["adaboost"],
    "Logistic Regression": ["logistic regression"],
    "Linear Regression": ["linear regression"],
    "Ridge Regression": ["ridge regression"],
    "Lasso Regression": ["lasso regression", "lasso"],
    "SVM": ["svm", "svms", "support vector machines?", "support vector classifier", "svr"],
    "KNN": ["knn", "k-nn", "k nearest neighbou?rs?", "k-nearest neighbou?rs?"],
    "Naive Bayes": ["naive bayes", "naïve bayes"],
    "K-Means": ["k-means", "kmeans", "k means"],
    "DBSCAN": ["dbscan"],
    "ARIMA": ["arima"],
    "SARIMA": ["sarima", "sarimax"],
    "Prophet": ["prophet"],
    "LSTM": ["lstm", "lstms"],
    "GRU": ["gru"],
    "CNN": ["cnn", "cnns", "convolutional neural networks?"],
    "RNN": ["rnn", "rnns", "recurrent neural networks?"],
    "Transformer": ["transformers?"],
    "BERT": ["bert", "roberta", "distilbert"],
    "Neural Network": ["neural networks?", "neural nets?", "mlp", "deep learning"],
}

METRICS: dict[str, list[str]] = {
    "RMSE": ["rmse", "root mean squared? error"],
    "MSE": ["mse", "mean squared? error"],
    "MAE": ["mae", "mean absolute error"],
    "MAPE": ["mape", "mean absolute percentage error"],
    "R-squared": ["r2", "r²", "r-squared", "r squared"],
    "Accuracy": ["accuracy"],
    "Precision": ["precision"],
    "Recall": ["recall"],
    "F1-score": ["f1", "f1-score", "f1 score", "f-score", "f score"],
    "ROC AUC": ["roc auc", "roc-auc", "auc", "auroc"],
    "Log Loss": ["log loss", "logloss", "cross-entropy"],
    "Silhouette Score": ["silhouette score", "silhouette"],
    "Cohen's Kappa": ["cohen'?s kappa", "cohen kappa", "kappa"],
}

TASK_TYPES: dict[str, list[str]] = {
    "classification": ["classification", "classify", "classifying", "classifier"],
    "regression": ["regression"],
    "forecasting": ["forecasting", "forecast", "time series", "time-series"],
    "clustering": ["clustering", "cluster", "segmentation"],
    "anomaly detection": ["anomaly detection", "outlier detection", "fraud detection"],
    "recommendation": ["recommendation", "recommender"],
}

DOMAINS: dict[str, list[str]] = {
    "tourism": ["tourism", "travel", "hospitality"],
    "healthcare": ["healthcare", "health care", "health", "medical", "medicine"],
    "finance": ["finance", "financial", "banking", "fintech", "insurance"],
    "retail": ["retail", "e-commerce", "ecommerce", "sales"],
    "education": ["education", "educational", "students"],
    "agriculture": ["agriculture", "farming", "crops?"],
    "energy": ["energy", "electricity", "power consumption"],
    "transport": ["transport", "transportation", "traffic", "logistics"],
    "sports": ["sports?", "football", "cricket"],
    "marketing": ["marketing", "advertising", "customer churn"],
    "telecom": ["telecom", "telecommunications"],
    "real estate": ["real estate", "housing", "property"],
    "environment": ["environment", "climate", "weather", "air quality"],
}

RESOURCES: dict[str, list[str]] = {
    "GPU": ["gpus?"],
    "TPU": ["tpus?"],
    "Google Colab": ["google colab", "colab"],
    "Kaggle notebooks": ["kaggle notebooks?", "kaggle kernels?"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "Google Cloud": ["gcp", "google cloud"],
}

HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "probably", "i think", "i guess", "i suppose",
    "not sure", "unsure", "might", "could we", "should we", "shall we", "what about",
    "how about", "what if", "not certain", "i'm thinking", "kind of", "sort of",
]

# A value named right after one of these is being rejected, not proposed.
NEGATION = re.compile(
    r"\b(?:not|no|never|don'?t|do not|doesn'?t|avoid|without|instead of|rather than|except|"
    r"drop|skip|scrap)\s+(?:use\s+|using\s+|the\s+|a\s+|an\s+)?(?:\w+\s+){0,1}$",
    re.IGNORECASE,
)

# Words that are not a real value when captured by a phrase pattern.
VAGUE = {
    "it", "this", "that", "them", "these", "those", "something", "anything", "what",
    "which", "one", "stuff", "things", "data", "the data", "our data", "this data", "more",
}

# "the training data" or "a lot of data" describe data, they do not name a dataset.
GENERIC_DATA_WORDS = {
    "training", "test", "testing", "validation", "raw", "more", "new", "same", "enough", "some",
    "real", "sample", "clean", "cleaned", "missing", "full", "whole", "other", "existing", "labelled",
    "labeled", "big", "small", "large", "good", "bad", "any", "all", "this", "that", "our", "my",
    # found in real project documents ("the main data needed is ...")
    "main", "primary", "project", "own", "team", "input", "required", "needed", "relevant", "available",
    "collected", "synthetic", "public", "private", "chat", "text", "planning", "historical", "entire",
    "final", "original", "tabular", "structured", "unstructured", "numeric", "numerical", "categorical",
    "outcome", "experimental", "such", "the", "their", "its", "your", "meta",
}

# Words that end a captured phrase ("predict churn USING xgboost").
_STOP = (
    r"using|with|from|based|by|because|so|since|but|and then|then|for the|instead|"
    r"which|that|as our|as the|or"
)
_PHRASE_VALUE = r"(?P<v>[A-Za-z0-9_][\w\-' /]{1,50}?)"
_PHRASE_END = rf"(?=\s*(?:[.,!?;:)]|\s+(?:{_STOP})\b|$))"

# Words at the edge of a captured phrase that carry no meaning
_EDGE_WORDS = re.compile(r"^(?:the|a|an|our|my|their|some)\s+|\s+(?:please|instead|too|also|then)$", re.IGNORECASE)

# Signs that a message is about planning, so it is worth asking the LLM when no rule matched.
PLANNING_CUES = re.compile(
    r"\b(?:predict\w*|forecast\w*|target|label|dataset|data ?set|data|model\w*|algorithm\w*|"
    r"method|metric\w*|evaluat\w*|measure|accuracy|problem|goal|objective|aim|domain|topic|"
    r"classif\w*|regress\w*|cluster\w*|feature\w*|train\w*|gpu|compute|budget|deadline|scope)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------- helpers
def _alternation(spellings: Iterable[str]) -> str:
    return "|".join(sorted(spellings, key=len, reverse=True))


def _compile(vocab: dict[str, list[str]]) -> list[tuple[str, re.Pattern]]:
    return [
        (canonical, re.compile(rf"(?<![\w-])(?:{_alternation(spellings)})(?![\w-])", re.IGNORECASE))
        for canonical, spellings in vocab.items()
    ]


_ALGORITHMS = _compile(ALGORITHMS)
_METRICS = _compile(METRICS)
_TASK_TYPES = _compile(TASK_TYPES)
_RESOURCES = _compile(RESOURCES)
_DOMAIN_WORDS = _alternation(s for spellings in DOMAINS.values() for s in spellings)
_HEDGE = re.compile(rf"(?<!\w)(?:{_alternation(HEDGE_WORDS)})(?!\w)", re.IGNORECASE)

_TARGET_PATTERNS = [
    re.compile(rf"\b(?:predict(?:ing|s|ed)?|forecast(?:ing|s)?|estimat(?:e|ing))\s+"
               rf"(?:the\s+)?{_PHRASE_VALUE}{_PHRASE_END}", re.IGNORECASE),
    re.compile(rf"\btarget(?:\s+(?:variable|column|label|group))?\s*(?:is|should be|will be|would be|could be|"
               rf"as|=|:)\s+(?:the\s+)?{_PHRASE_VALUE}{_PHRASE_END}", re.IGNORECASE),
]
_DATASET_PATTERNS = [
    re.compile(rf"\b(?:data ?set|data)\s*(?:is|will be|should be|=|:)\s+(?:the\s+)?{_PHRASE_VALUE}{_PHRASE_END}",
               re.IGNORECASE),
    # "the Kaggle tourism dataset", "the PIPES meta-dataset", "the AMI Meeting Corpus"
    re.compile(r"\b(?:the|a|an|this|that|our)\s+(?P<v>[A-Za-z0-9][\w\-' ]{0,40}?(?:\s+|\s*-)"
               r"(?:meta-?)?(?:data ?set|data|corpus))\b",
               re.IGNORECASE),
]
_DOMAIN_PATTERNS = [
    re.compile(rf"\b(?:domain|sector|industry|field|area|topic|group|project|theme)\s*(?:is|will be|should be|"
               rf"=|:)\s+(?:about\s+|on\s+|in\s+)?(?:the\s+)?(?P<v>{_DOMAIN_WORDS})\b", re.IGNORECASE),
    re.compile(rf"\b(?:in|for|on)\s+(?:the\s+)?(?P<v>{_DOMAIN_WORDS})\s+(?:domain|sector|industry|field|area)\b",
               re.IGNORECASE),
]
_PROBLEM_PATTERNS = [
    re.compile(rf"\b(?:the\s+)?problem\s+(?:is|will be|we(?:'re| are) solving is)\s+(?:that\s+|how\s+to\s+)?"
               rf"{_PHRASE_VALUE}{_PHRASE_END}", re.IGNORECASE),
]
_OBJECTIVE_PATTERNS = [
    re.compile(rf"\b(?:our\s+|the\s+|main\s+)?(?:goal|objective|aim)\s+(?:is|will be|should be)\s+(?:to\s+)?"
               rf"{_PHRASE_VALUE}{_PHRASE_END}", re.IGNORECASE),
]


def _rejected(message: str, start: int) -> bool:
    return bool(NEGATION.search(message[max(0, start - 30):start]))


def _overlaps(start: int, end: int, taken: list[tuple[int, int]]) -> bool:
    return any(start < t_end and t_start < end for t_start, t_end in taken)


def _clean(value: str) -> Optional[str]:
    v = " ".join(value.split()).strip(" -'/")
    previous = None
    while previous != v:
        previous = v
        v = _EDGE_WORDS.sub("", v).strip(" -'/")
    if len(v) < 2 or v.lower() in VAGUE:
        return None
    return v


def _names_a_dataset(value: str) -> bool:
    words = re.sub(r"(?:\s+|\s*-)(?:meta-?)?(?:data ?set|data|corpus)$", "", value, flags=re.IGNORECASE).lower().split()
    return bool(words) and " of " not in f" {' '.join(words)} " and not all(w in GENERIC_DATA_WORDS for w in words)


def _domain_canonical(word: str) -> str:
    for canonical, spellings in DOMAINS.items():
        if re.fullmatch(_alternation(spellings), word, re.IGNORECASE):
            return canonical
    return word.lower()


def _vocab_hits(message, field, compiled, taken):
    """(start, end, field, value) for each word-list match not rejected or inside an earlier match."""
    hits = []
    for canonical, pattern in compiled:
        for m in pattern.finditer(message):
            if _overlaps(m.start(), m.end(), taken) or _rejected(message, m.start()):
                continue
            hits.append((m.start(), m.end(), field, canonical))
    return hits


def _pattern_hits(message, field, patterns, taken, canonical=None):
    hits = []
    for pattern in patterns:
        for m in pattern.finditer(message):
            value = _clean(m.group("v"))
            if value is None or _rejected(message, m.start()):
                continue
            # The span is the whole phrase ("predict tourist arrivals"), so the reader sees the evidence.
            if _overlaps(m.start("v"), m.end("v"), taken):
                continue
            hits.append((m.start(), m.end(), field, canonical(value) if canonical else value))
    return hits


# ------------------------------------------------------------------------ main
def rule_items(message: str) -> list[ExtractedItem]:
    """Extract planning facts from one message with rules only. Never raises on ordinary text."""
    if not message or not message.strip():
        return []
    hedged = bool(_HEDGE.search(message)) or message.rstrip().endswith("?")

    # Algorithms first, so "logistic regression" is not also read as the task "regression".
    taken: list[tuple[int, int]] = []
    hits = _vocab_hits(message, FieldName.ALGORITHM, _ALGORITHMS, taken)
    taken += [(s, e) for s, e, *_ in hits]
    for field, compiled in ((FieldName.EVALUATION_METRIC, _METRICS), (FieldName.RESOURCES, _RESOURCES)):
        new = _vocab_hits(message, field, compiled, taken)
        taken += [(s, e) for s, e, *_ in new]
        hits += new
    hits += _vocab_hits(message, FieldName.TASK_TYPE, _TASK_TYPES, taken)

    hits += _pattern_hits(message, FieldName.TARGET_VARIABLE, _TARGET_PATTERNS, taken)
    hits += [h for h in _pattern_hits(message, FieldName.DATASET, _DATASET_PATTERNS, taken) if _names_a_dataset(h[3])]
    hits += _pattern_hits(message, FieldName.DOMAIN, _DOMAIN_PATTERNS, [], canonical=_domain_canonical)
    hits += _pattern_hits(message, FieldName.PROBLEM, _PROBLEM_PATTERNS, taken)
    hits += _pattern_hits(message, FieldName.OBJECTIVE, _OBJECTIVE_PATTERNS, taken)

    # Dataset size: "10k rows" -> "10000 rows"
    for words, rows in find_sizes(message):
        start = message.find(words)
        if not _rejected(message, start):
            hits.append((start, start + len(words), FieldName.DATASET_SIZE, f"{rows} rows"))

    # Timeline: the overall deadline or length is one value like any field; milestones are extra items.
    overall, milestones = find_timeline(message)
    for found in overall:
        start = message.find(found.span)
        if not _rejected(message, start):
            hits.append((start, start + len(found.span), FieldName.TIMELINE, found.value))

    # One value per field per message, in the order they appear.
    by_field: dict[FieldName, list[tuple[int, int, str]]] = {}
    for start, end, field, value in sorted(hits, key=lambda h: h[0]):
        by_field.setdefault(field, []).append((start, end, value))

    items: list[ExtractedItem] = []
    for field in FieldName:
        found = by_field.get(field)
        if not found:
            continue
        start, end, value = found[0]
        several = len({v.lower() for _, _, v in found}) > 1
        items.append(ExtractedItem(field=field, value=value, span=message[start:end], hedged=hedged or several))
    seen = set()
    for found in milestones:                     # each named milestone once, even if repeated
        if found.value.lower() not in seen:
            seen.add(found.value.lower())
            items.append(ExtractedItem(field=FieldName.TIMELINE, value=found.value, span=found.span, hedged=hedged))
    return items


_CANONICAL = {"algorithm": _ALGORITHMS, "metric": _METRICS, "task": _TASK_TYPES, "resource": _RESOURCES}


def canonical_names(kind: str, text: Optional[str]) -> list[str]:
    """The known names in a value, in standard spelling: canonical_names("algorithm", "an xgboost model")
    -> ["XGBoost"]. kind is algorithm, metric, task or resource. Empty if none is recognised."""
    if not text:
        return []
    found, taken = [], []
    if kind == "task":       # "logistic regression" is an algorithm, not the task "regression"
        taken = [(m.start(), m.end()) for _, p in _ALGORITHMS for m in p.finditer(text)]
    for canonical, pattern in _CANONICAL[kind]:
        for m in pattern.finditer(text):
            if not _overlaps(m.start(), m.end(), taken):
                found.append((m.start(), canonical))
                break
    return [name for _, name in sorted(found)]


def looks_like_planning(message: str) -> bool:
    """True if the message mentions planning words, so asking the LLM is worth the wait."""
    return bool(PLANNING_CUES.search(message or ""))
