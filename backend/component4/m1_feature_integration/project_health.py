"""Project-health signals: reproducibility, data quality and model performance vs target.

Reproducibility and the data-validation bonus come from repo file names (never contents), data quality
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


def reproducibility(files: list[str]) -> float | None:
    if not files:
        return None
    return round(check(SETTINGS["reproducibility"], normalise(files))["share"], 3)


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


def target(metric_text: str | None) -> float | None:
    """The group's target on a 0-1 scale; None when the metric is an error (lower is better)."""
    text = (metric_text or "").lower()
    if any(word in text for word in SETTINGS["model_performance"]["lower_is_better"]):
        return None
    match = TARGET_NUMBER.search(text)
    if match and match.group(1):
        return float(match.group(1))
    if match and match.group(2) and 0 < float(match.group(2)) <= 100:
        return float(match.group(2)) / 100
    return SETTINGS["model_performance"]["default_target"]


def model_performance(runs: list[dict] | None, metric_text: str | None) -> float | None:
    fields = SETTINGS["model_performance"]["score_fields"]
    scores = [run[field] for run in runs or [] for field in fields if run.get(field) is not None]
    goal = target(metric_text)
    if not scores or not goal:
        return None
    return round(min(max(scores) / goal, 1.0), 3)


def features(runs: list[dict] | None, files: list[str], metric_text: str | None) -> dict:
    return {"reproducibility": reproducibility(files), "data_quality": data_quality(runs, files),
            "model_performance": model_performance(runs, metric_text)}
