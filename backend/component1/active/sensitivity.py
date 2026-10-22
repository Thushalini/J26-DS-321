"""Which unknown fact decides whether the plan is feasible?

    deps = feasibility_dependencies(feasibility_bundle, state, events)
    deps[0]["text"] == "whether the plan is feasible depends on the dataset size: it looks infeasible with
                        about 500 rows, risky with about 5,000 and feasible with about 50,000"

Only facts the team has NOT stated, or stated unsurely, are tried: the dataset size, a GPU and the
time left. Each one is set to a few likely values (the others stay as they are) and the feasibility
model (feasibility/predict.py, unchanged) judges every version. A fact is named only when the verdict
changes with it; when it does not, nothing is said. Facts stated clearly are never second-guessed.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

from ..common.schema import FieldName, FieldStatus
from ..extraction.rules import canonical_names
from ..extraction.size import rows_of
from ..feasibility.predict import COMPUTE, judge
from ..timeline.current import current_timeline

ROWS = (500, 5_000, 50_000)
WEEKS = (2, 6, 12)
DEFAULT_ROWS = 5_000          # stands in for an unknown size while another fact is being tried
SURE = (FieldStatus.KNOWN, FieldStatus.CONFIRMED)
PREFIX = "whether the plan is feasible depends on"


def _joined(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _state_of(state, name: FieldName) -> str:
    status = state.fields[name].status
    return "sure" if status in SURE else "unsure" if status == FieldStatus.UNCERTAIN else "missing"


def feasibility_dependencies(bundle: Optional[dict], state, events: Iterable = (),
                             today: Optional[date] = None) -> list[dict]:
    """The facts whose value would change the feasibility verdict, each with what happens per value."""
    if bundle is None:
        return []
    algorithm = state.fields[FieldName.ALGORITHM]
    if algorithm.status not in SURE + (FieldStatus.UNCERTAIN,) or not algorithm.value:
        return []
    task = state.fields[FieldName.TASK_TYPE].value if _state_of(state, FieldName.TASK_TYPE) != "missing" else ""

    size = _state_of(state, FieldName.DATASET_SIZE)
    rows = rows_of(state.fields[FieldName.DATASET_SIZE].value) if size != "missing" else None
    compute = _state_of(state, FieldName.RESOURCES)
    gpu = bool(set(canonical_names("resource", state.fields[FieldName.RESOURCES].value)) & COMPUTE)
    timeline = _state_of(state, FieldName.TIMELINE)
    weeks = None
    if timeline != "missing":
        t = current_timeline(state, list(events), today)
        weeks = max(0.0, t["days_left"] / 7) if t and t.get("days_left") is not None else None
        if weeks is None:
            timeline = "missing"                     # said, but no date could be worked out from it

    base = {"task": task or "", "algorithm": algorithm.value, "rows": rows or DEFAULT_ROWS,
            "has_gpu": gpu, "weeks_left": weeks}
    tries = []
    if size != "sure" or not rows:
        values = sorted(set(ROWS) | ({rows} if rows else set()))
        tries.append(("dataset_size", "the dataset size", size, "rows", values))
    if compute != "sure":
        tries.append(("resources", "having a GPU", compute, "has_gpu", [False, True]))
    if timeline != "sure":
        values = sorted(set(WEEKS) | ({round(weeks)} if weeks is not None else set()))
        tries.append(("timeline", "the time left", timeline, "weeks_left", values))

    out = []
    for field, what, known, key, values in tries:
        scenarios = []
        for v in values:
            j = judge(bundle, **{**base, key: v})
            if j is None:
                break
            scenarios.append({"value": v, "label": j["label"], "feasible_probability": j["probabilities"]["feasible"]})
        if len(scenarios) < len(values) or len({s["label"] for s in scenarios}) < 2:
            continue                                  # the verdict does not change: nothing to say
        out.append({"field": field, "fact": what, "known": known, "scenarios": scenarios,
                    "text": f"{PREFIX} {what}: it looks {_describe(key, scenarios)}"})
    return out


def _describe(key: str, scenarios: list[dict]) -> str:
    """e.g. "infeasible with about 500 rows and risky with about 5,000 or more": values in a row that
    give the same verdict are said once."""
    if key == "has_gpu":
        by = {s["value"]: s["label"] for s in scenarios}
        return f"{by[False]} without a GPU but {by[True]} with one"
    runs: list[list[dict]] = []
    for s in scenarios:
        if runs and runs[-1][-1]["label"] == s["label"]:
            runs[-1].append(s)
        else:
            runs.append([s])
    about, unit = ("about ", " rows") if key == "rows" else ("", " weeks left")
    parts = []
    for i, run in enumerate(runs):
        first, last = f"{run[0]['value']:,}", f"{run[-1]['value']:,}"
        if len(run) == 1:
            amount = first
        elif i == len(runs) - 1:
            amount = f"{first} or more"
        else:
            amount = f"{first} to {last}"
        parts.append(f"{run[0]['label']} with {about}{amount}{unit if i == 0 else ''}")
    return _joined(parts)
