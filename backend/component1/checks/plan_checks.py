"""Run the plan checks on a blueprint, store the results, and tell Buddy about problems.

    results = run_checks(state, events)       # read only: a list of CheckResult
    results = record_checks(store)            # also stores {check: pass/warning/fail} when it changed
    signals = check_signals(results)          # one CHECK signal per warning or fail

A check runs only when the fields it needs have a value the team is working with (known,
confirmed or uncertain). A field the team still disagrees about is skipped, because there is no
single plan to check yet. The stored results are the blueprint's `feasibility`, which
Component 4 turns into its feasibility score (pass 1.0, warning 0.5, fail 0.0).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from ..agent.signals import CHECK, Signal
from ..common.events import EventStore
from ..common.schema import FieldName, FieldStatus
from ..extraction.rules import canonical_names
from ..timeline.current import current_timeline
from . import knowledge as K

ACTOR = "plan_checker"
WORKING = (FieldStatus.KNOWN, FieldStatus.CONFIRMED, FieldStatus.UNCERTAIN)
LABELS = {
    "metric_fits_task": "Metric fits the task",
    "algorithm_fits_task": "Algorithm fits the task",
    "target_fits_task": "Target fits the task",
    "compute_fits_model": "Compute fits the model",
    "timeline_fits_model": "Time fits the model",
}


@dataclass(frozen=True)
class CheckResult:
    key: str                         # one of LABELS
    field: str                       # the field a team would change to fix it
    status: str                      # pass / warning / fail
    detail: str                      # why, in a sentence (empty for a pass)
    alternatives: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return LABELS[self.key]

    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "field": self.field, "status": self.status,
                "detail": self.detail, "alternatives": list(self.alternatives)}


def _value(state, name: FieldName) -> Optional[str]:
    f = state.fields[name]
    return f.value if f.status in WORKING else None


def _first(kind: str, text: Optional[str]) -> Optional[str]:
    names = canonical_names(kind, text)
    return names[0] if names else None


def _or(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _has_word(text: str, words) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text) for w in words)


def run_checks(state, events: Iterable = (), today: Optional[date] = None) -> list[CheckResult]:
    task = _first("task", _value(state, FieldName.TASK_TYPE))
    metric = _first("metric", _value(state, FieldName.EVALUATION_METRIC))
    algorithm = _first("algorithm", _value(state, FieldName.ALGORITHM))
    target = _value(state, FieldName.TARGET_VARIABLE)
    results: list[CheckResult] = []

    if task in K.TASK_METRICS and metric:
        if metric in K.TASK_METRICS[task]:
            results.append(CheckResult("metric_fits_task", "evaluation_metric", "pass", ""))
        else:
            alts = tuple(K.METRIC_ALTERNATIVES[task])
            kind = K.METRIC_FAMILY.get(metric, "not usually used for this")
            results.append(CheckResult(
                "metric_fits_task", "evaluation_metric", "fail",
                f"{metric} is {kind}, but the task is {task}. Consider {_or(alts)} instead.", alts))

    if task in K.TASK_ALGORITHMS and algorithm:
        if algorithm in K.TASK_ALGORITHMS[task]:
            results.append(CheckResult("algorithm_fits_task", "algorithm", "pass", ""))
        else:
            alts = tuple(K.ALGORITHM_ALTERNATIVES[task])
            kind = K.ALGORITHM_FAMILY.get(algorithm)
            why = f"{algorithm} is {kind}" if kind else f"{algorithm} is not usually used for {task}"
            results.append(CheckResult(
                "algorithm_fits_task", "algorithm", "fail",
                f"{why}, but the task is {task}. Consider {_or(alts)} instead.", alts))

    if task and target:
        low = target.lower()
        number, category = _has_word(low, K.NUMBER_TARGET_WORDS), _has_word(low, K.CATEGORY_TARGET_WORDS)
        if task in K.CATEGORY_TASKS and number and not category:
            results.append(CheckResult(
                "target_fits_task", "task_type", "warning",
                f"'{target}' sounds like a number, so this may be regression (or forecasting, if it is over time) "
                f"rather than {task}.", ("regression", "forecasting")))
        elif task in K.NUMBER_TASKS and category and not number:
            results.append(CheckResult(
                "target_fits_task", "task_type", "warning",
                f"'{target}' sounds like a category, so this may be classification rather than {task}.",
                ("classification",)))
        elif number or category:
            results.append(CheckResult("target_fits_task", "task_type", "pass", ""))

    if algorithm in K.DEEP_LEARNING:
        baselines = tuple(K.SIMPLER_BASELINES.get(task or "", ["XGBoost"]))
        compute = set(canonical_names("resource", _value(state, FieldName.RESOURCES)))
        if compute & K.COMPUTE_RESOURCES:
            results.append(CheckResult("compute_fits_model", "resources", "pass", ""))
        else:
            results.append(CheckResult(
                "compute_fits_model", "resources", "warning",
                f"{algorithm} is a deep learning model, which usually needs a GPU, and no GPU or cloud compute "
                f"has been mentioned. Google Colab has a free GPU; or consider {_or(baselines)}.",
                ("Google Colab",) + baselines))

        t = current_timeline(state, events, today) if state.fields[FieldName.TIMELINE].status in WORKING else None
        if t and t.get("days_left") is not None:
            days = t["days_left"]
            if days < K.SHORT_TIMELINE_DAYS:
                left = "the deadline has passed" if days < 0 else f"only {days} days are left"
                results.append(CheckResult(
                    "timeline_fits_model", "algorithm", "warning",
                    f"{left}, which is little time to train and tune {algorithm}. "
                    f"Consider a simpler baseline such as {_or(baselines)} first.", baselines))
            else:
                results.append(CheckResult("timeline_fits_model", "algorithm", "pass", ""))
    return results


def feasibility_of(results: Iterable[CheckResult]) -> dict[str, str]:
    """{check: pass/warning/fail}, the shape the blueprint's `feasibility` and Component 4 use."""
    return {r.key: r.status for r in results}


def check_signals(results: Iterable[CheckResult]) -> list[Signal]:
    """What Buddy should be told: one CHECK per problem. The detail is part of what Buddy remembers,
    so the same problem is not repeated, but a different one (or the same one later) is."""
    return [Signal(CHECK, r.field, r.alternatives, detail=r.detail) for r in results if r.status != "pass"]


def record_checks(store: EventStore, today: Optional[date] = None) -> list[CheckResult]:
    """Run the checks and store the results in the blueprint when they changed."""
    results = run_checks(store.state, store.events, today)
    now = feasibility_of(results)
    before = store.state.feasibility or {}
    if now != before:
        store.update_scores(actor=ACTOR, feasibility=now)
    return results
