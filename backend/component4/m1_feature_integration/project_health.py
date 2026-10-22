"""Project-health signals: reproducibility, data quality and model performance vs target.

Reproducibility comes from the repo (key file contents when a scan exists, else file names), data quality
from Component 3's data-leakage probabilities, and performance from Component 3's best validation score
against the group's target (a number in the blueprint's metric, else the default). Each one stays
missing when its evidence is missing. No file counts towards both these and the readiness rubric.
"""
import re
from statistics import mean

from ..common import config
from ..m4_readiness.rubric import check, first_match, normalise

SETTINGS = config.PROJECT_HEALTH
TARGET_NUMBER = re.compile(r"(?<![\w.])(0?\.\d+|1(?:\.0+)?)(?![\w.])|(\d{1,3}(?:\.\d+)?)\s*%")
NUMBER = r"(0?\.\d+|1(?:\.0+)?|\d{1,3}(?:\.\d+)?\s*%)"
SCORE_TARGET = re.compile(r"(?:accuracy|f1|f-1|auc|roc|precision|recall|r2|r²|iou|bleu|dice)\b[^.\n]{0,25}?" + NUMBER
                          + r"|(?:at least|above|over|reach|target of|>=|≥)\s*" + NUMBER)


def reproducibility(files: list[str], evidence: dict | None = None) -> float | None:
    if not files:
        return None
    return round(check(SETTINGS["reproducibility"], normalise(files), evidence)["share"], 3)


def data_quality(runs: list[dict] | None, files: list[str]) -> float | None:
    cause = SETTINGS["data_quality"]["cause"]
    leakage = [run["root_cause_probabilities"][cause] for run in runs or []
               if cause in (run.get("root_cause_probabilities") or {})]
    if not leakage:
        return None
    score = 1 - mean(leakage)
    if files and first_match(SETTINGS["data_quality"]["validation_patterns"], normalise(files)):
        score += SETTINGS["data_quality"]["validation_bonus"]
    return round(min(score, 1.0), 3)


def _share(token: str) -> float | None:
    token = token.strip()
    value = float(token.rstrip("%").strip())
    if token.endswith("%"):
        return value / 100 if 0 < value <= 100 else None
    return value if 0 < value <= 1 else None


def target(metric_text: str | None, objective_text: str | None = None) -> float | None:
    """The group's target on a 0-1 scale: a number in the metric, else a score target in the objective
    ("85% accuracy", "F1 above 0.8"), else the default. None when the metric is an error (lower is better)."""
    text = (metric_text or "").lower()
    if any(word in text for word in SETTINGS["model_performance"]["lower_is_better"]):
        return None
    match = TARGET_NUMBER.search(text)
    if match and match.group(1):
        return float(match.group(1))
    if match and match.group(2) and 0 < float(match.group(2)) <= 100:
        return float(match.group(2)) / 100
    goal = SCORE_TARGET.search((objective_text or "").lower())
    found = goal and _share(goal.group(1) or goal.group(2))
    return found or SETTINGS["model_performance"]["default_target"]


def model_performance(runs: list[dict] | None, metric_text: str | None, objective_text: str | None = None) -> float | None:
    fields = SETTINGS["model_performance"]["score_fields"]
    scores = [run[field] for run in runs or [] for field in fields if run.get(field) is not None]
    goal = target(metric_text, objective_text)
    if not scores or not goal:
        return None
    return round(min(max(scores) / goal, 1.0), 3)


def features(runs: list[dict] | None, files: list[str], metric_text: str | None, evidence: dict | None = None,
             objective_text: str | None = None) -> dict:
    return {"reproducibility": reproducibility(files, evidence), "data_quality": data_quality(runs, files),
            "model_performance": model_performance(runs, metric_text, objective_text)}
