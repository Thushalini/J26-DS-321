"""Is a value specific enough to build on? If not, the question that would make it specific.

    clarify_question("algorithm", "Neural Network")   # "The algorithm is 'Neural Network'. Which model exactly? ..."
    clarify_question("algorithm", "XGBoost")          # None: a named model is specific

Rules, built on the vocabulary Buddy already has (extraction/rules.py): a known name is specific,
except the two family names that stand for many models ("Neural Network", which is also what
"deep learning" becomes, and "Transformer"). A free-text value is vague when every word in it is a
generic word ("Kaggle data", "the output", "prediction"). A timeline is vague when no date, duration
or week can be read from it ("soon"). Today such values are either thrown away (rules.VAGUE) or
saved silently; this keeps them and asks for the missing detail instead.
"""
from __future__ import annotations

import re
from typing import Optional

from ..extraction.rules import ALGORITHMS, DOMAINS, GENERIC_DATA_WORDS, METRICS, RESOURCES, TASK_TYPES, VAGUE

FAMILIES = {
    "Neural Network": "an MLP, a CNN or an LSTM",
    "Transformer": "BERT or another named model",
}
ALGORITHM_WORDS = {
    "machine", "learning", "ml", "ai", "model", "models", "algorithm", "algorithms", "ensemble", "tree", "trees",
    "based", "deep", "supervised", "unsupervised", "statistical", "classical", "classifier", "regressor",
    "approach", "method", "technique", "simple", "basic", "standard", "best", "good", "some", "kind", "of",
}
METRIC_WORDS = {"performance", "results", "result", "quality", "error", "score", "scores", "metric", "metrics",
                "evaluation", "how", "well", "good", "best", "overall", "some", "kind", "of"}
TASK_WORDS = {"prediction", "predict", "predicting", "predictive", "modelling", "modeling", "analysis",
              "analytics", "machine", "learning", "ml", "ai", "model", "data", "science", "insights", "task"}
DATA_WORDS = GENERIC_DATA_WORDS | {"data", "dataset", "datasets", "set", "sets", "a", "an", "from", "of", "some",
                                   "on", "online", "internet", "web", "open", "government"}
SOURCES = {"kaggle": "Kaggle", "uci": "UCI", "github": "GitHub", "huggingface": "Hugging Face"}
TARGET_WORDS = {"output", "outputs", "result", "results", "label", "labels", "class", "classes", "target",
                "value", "values", "outcome", "outcomes", "prediction", "predictions", "y", "answer", "score",
                "number", "variable", "column", "something", "the", "a", "an", "it", "final", "main"}
DOMAIN_WORDS = {"data", "science", "ai", "machine", "learning", "business", "general", "technology", "tech",
                "it", "computer", "industry", "real", "world", "many", "various", "any"}
PLAN_WORDS = {"predict", "prediction", "predicting", "analysis", "analyse", "analyze", "analysing", "data",
              "model", "models", "build", "building", "make", "making", "create", "a", "an", "the", "ml", "ai",
              "machine", "learning", "project", "system", "app", "something", "improve", "better", "insights",
              "accuracy", "good", "results", "solve", "problem", "do", "some", "to"}
RESOURCE_WORDS = {"cloud", "computer", "computers", "server", "servers", "online", "machine", "machines",
                  "something", "a", "the", "some", "our", "my", "good", "powerful"}
# Small words that never make a value specific; allowed in every field's list.
FUNCTION_WORDS = {"a", "an", "the", "it", "its", "is", "are", "be", "do", "does", "doing", "get", "getting", "some",
                  "of", "for", "to", "on", "in", "with", "and", "or", "our", "my", "we", "us", "how", "what", "kind"}
_WORD = re.compile(r"[a-z0-9]+")


def _words(value: str) -> list[str]:
    return _WORD.findall((value or "").lower())


def _only(words: list[str], allowed: set[str]) -> bool:
    """Every word is generic (and at least one is not just a small function word)."""
    return (bool(words) and all(w in allowed or w in FUNCTION_WORDS for w in words)
            and any(w not in FUNCTION_WORDS for w in words))


def _metric_examples(task: Optional[str]) -> str:
    return {"regression": "RMSE or MAE", "forecasting": "RMSE or MAPE", "classification": "F1-score or accuracy",
            "clustering": "the silhouette score"}.get((task or "").lower(), "RMSE, MAE, F1-score or accuracy")


def clarify_question(field: str, value: Optional[str], task: Optional[str] = None) -> Optional[str]:
    """The question that would make the value specific, or None if it is specific enough already."""
    if not value or not value.strip():
        return None
    words = _words(value)
    if value.strip().lower() in VAGUE:
        return f"The {field.replace('_', ' ')} is '{value}'. Could someone say exactly what it is?"

    if field == "algorithm":
        if value in FAMILIES:
            return f"The algorithm is '{value}'. Which model exactly? For example {FAMILIES[value]}."
        if value not in ALGORITHMS and _only(words, ALGORITHM_WORDS):
            return (f"The algorithm is '{value}'. Which model exactly? "
                    f"For example XGBoost, Random Forest or Logistic Regression.")
    elif field == "evaluation_metric":
        if value not in METRICS and _only(words, METRIC_WORDS):
            return f"The evaluation metric is '{value}'. Which metric exactly? For example {_metric_examples(task)}."
    elif field == "task_type":
        if value not in TASK_TYPES and _only(words, TASK_WORDS):
            return f"The task is '{value}'. Is it classification, regression, forecasting or clustering?"
    elif field == "dataset":
        if _only(words, DATA_WORDS | set(SOURCES)):
            source = next((SOURCES[w] for w in words if w in SOURCES), None)
            which = f"Which {source} dataset" if source else "Which dataset"
            return f"The dataset is '{value}'. {which} exactly? A name or a link helps."
    elif field == "target_variable":
        if _only(words, TARGET_WORDS):
            return f"The target is '{value}'. Which column or quantity exactly will the model predict?"
    elif field == "domain":
        if value not in DOMAINS and _only(words, DOMAIN_WORDS):
            return f"The domain is '{value}'. Which area exactly? For example tourism, healthcare or finance."
    elif field in ("problem", "objective"):
        if len(words) <= 4 and _only(words, PLAN_WORDS):
            return (f"The {field} '{value}' is quite general. Could someone say it in one sentence: "
                    f"what, for whom, and why?")
    elif field == "resources":
        if value not in RESOURCES and _only(words, RESOURCE_WORDS):
            return ("Which compute exactly? For example a GPU, Google Colab, Kaggle notebooks "
                    "or a cloud provider such as AWS.")
    elif field == "timeline":
        from ..timeline.dates import resolve
        r = resolve(value)
        if (r is None or r["kind"] == "text") and not re.search(r"\d", value) and not value.lower().startswith("milestone"):
            return f"The timeline is '{value}'. When exactly is the deadline? A date or a number of weeks helps."
    return None


def vague_fields(state) -> dict[str, str]:
    """field -> clarifying question, for every field whose current value is too vague to build on."""
    from ..common.schema import FieldName, FieldStatus

    task = state.fields[FieldName.TASK_TYPE].value
    out = {}
    for name in FieldName:
        f = state.fields[name]
        if f.status in (FieldStatus.MISSING, FieldStatus.CONFLICTING):
            continue                     # missing: a gap question; conflicting: Buddy asks which value
        q = clarify_question(name.value, f.value, task)
        if q:
            out[name.value] = q
    return out
