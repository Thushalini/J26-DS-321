"""Use the saved feasibility classifier on one plan: verdict, reasons, and changes that would help.

    bundle = load_model()
    j = judge(bundle, task="classification", algorithm="LSTM", rows=2000, has_gpu=False, weeks_left=3)
    j["label"], j["probability"], j["reasons"], j["alternatives"]

    j = judge_blueprint(bundle, state, events)          # the same, read from a blueprint (or None)

Reasons come from SHAP (Lundberg and Lee, 2017): how much each part of the plan pushed the
model towards or away from "feasible". Alternatives are a counterfactual search in the spirit of
DiCE (Mothilal et al., 2020): every realistic change (another algorithm family for the same task,
or a GPU) is tried, and the ones that make the plan more feasible are kept, best first. The set of
realistic changes is small, so trying all of them is exact. "More weeks" is never suggested,
because a team cannot choose that.

The probabilities are the model's own and NOT calibrated, and the model was trained on simulated
outcomes (simulate.py). Say so wherever they are shown.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from ..common.schema import FieldName, FieldStatus
from ..extraction.rules import canonical_names
from ..extraction.size import rows_of
from ..timeline.current import current_timeline
from .simulate import ALGORITHM_FAMILY, FEATURE_NAMES, TASK_FAMILIES, TASKS, features

log = logging.getLogger("component1")

MODEL_PATH = Path(__file__).parent / "models" / "feasibility_classifier.joblib"
YES = {"yes", "y", "true", "1", "gpu"}
WORKING = (FieldStatus.KNOWN, FieldStatus.CONFIRMED, FieldStatus.UNCERTAIN)
COMPUTE = {"GPU", "TPU", "Google Colab", "Kaggle notebooks", "AWS", "Azure", "Google Cloud"}
RANK = {"feasible": 0, "risky": 1, "infeasible": 2}
MIN_GAIN = 0.10                       # an alternative must raise P(feasible) by at least this much

# The algorithm suggested for each family and task, when the search tries "another family".
FAMILY_EXAMPLE = {
    "linear": {"classification": "Logistic Regression", "regression": "Linear Regression",
               "forecasting": "Linear Regression"},
    "tree": {"*": "XGBoost"},
    "distance": {"*": "SVM"},
    "deep": {"forecasting": "LSTM", "*": "Neural Network"},
    "time_series": {"*": "ARIMA"},
    "clustering": {"*": "K-Means"},
}
FAMILY_WORDS = {"linear": "a linear model", "tree": "a tree-based model", "distance": "a kernel / distance model",
                "deep": "a deep learning model", "time_series": "a time-series model", "clustering": "a clustering model"}


def load_model(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def _weeks(value) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _family_of(algorithm: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    names = canonical_names("algorithm", algorithm) if algorithm else []
    name = names[0] if names else None
    return name, ALGORITHM_FAMILY.get(name) if name else None


def _task_of(task: Optional[str], family: Optional[str]) -> Optional[str]:
    names = canonical_names("task", task) if task else []
    name = names[0] if names else (task or "").strip().lower() or None
    if name in TASKS:
        return name
    # The algorithm alone can say the task for time series and clustering.
    return {"time_series": "forecasting", "clustering": "clustering"}.get(family or "")


def plan_features_from_values(task: str, algorithm: str, rows, gpu="", weeks_left=None) -> Optional[list[float]]:
    """The model's input from plain values, or None if task, algorithm or size cannot be read."""
    _, family = _family_of(algorithm)
    task_name = _task_of(task, family)
    n_rows = rows if isinstance(rows, int) else rows_of(str(rows))
    if task_name is None or family is None or not n_rows:
        return None
    has_gpu = gpu if isinstance(gpu, bool) else str(gpu).strip().lower() in YES
    return features(task_name, family, n_rows, has_gpu, _weeks(weeks_left))


# ---------------------------------------------------------------- reasons
_explainers: dict[int, object] = {}


def _shap_for_feasible(bundle: dict, x: np.ndarray) -> Optional[np.ndarray]:
    """SHAP values towards "feasible" for one plan, or None if SHAP is not available."""
    try:
        import shap
    except ImportError:
        return None
    model = bundle["model"]
    key = id(model)
    if key not in _explainers:
        _explainers[key] = shap.TreeExplainer(model)
    values = np.array(_explainers[key].shap_values(x.reshape(1, -1)))
    k = list(model.classes_).index("feasible")
    if values.ndim == 3:                       # (rows, features, classes): newer SHAP
        return values[0, :, k]
    return values[k][0]                        # [classes][rows, features]: older SHAP


def _reason_text(name: str, x: np.ndarray, inputs: dict) -> Optional[str]:
    value = x[FEATURE_NAMES.index(name)]
    if name == "log10_rows":
        return f"dataset size: {inputs['rows']:,} rows"
    if name.startswith("family_"):
        return f"{FAMILY_WORDS[name[7:]]} ({inputs['algorithm']})" if value == 1 else None
    if name.startswith("task_"):
        return f"task: {name[5:]}" if value == 1 else None
    if name == "has_gpu":
        return "GPU available" if value == 1 else "no GPU mentioned"
    if name == "weeks_left":
        return f"{int(inputs['weeks_left'])} weeks left" if inputs["weeks_left"] is not None else None
    if name == "timeline_known":
        return None if value == 1 else "timeline not stated yet"
    return None


def _reasons(bundle: dict, x: np.ndarray, inputs: dict, label: str, top: int = 3) -> list[dict]:
    """The parts of the plan that most support the verdict: what raises the risk of a risky or
    infeasible plan, or what lowers the risk of a feasible one. Parts pulling the other way are left
    out, because "2,000 rows lowers risk" next to "infeasible" only confuses people."""
    contributions = _shap_for_feasible(bundle, x)
    if contributions is None:
        return []
    wanted_sign = 1 if label == "feasible" else -1
    order = np.argsort(-np.abs(contributions))
    out = []
    for i in order:
        text = _reason_text(FEATURE_NAMES[i], x, inputs)
        if text is None or abs(contributions[i]) < 0.01 or np.sign(contributions[i]) != wanted_sign:
            continue
        out.append({"text": text, "effect": "lowers risk" if contributions[i] > 0 else "raises risk",
                    "weight": round(float(contributions[i]), 3)})
        if len(out) == top:
            break
    return out


# ------------------------------------------------------------ alternatives
def _predict(bundle: dict, x) -> dict[str, float]:
    p = bundle["model"].predict_proba(np.array(x).reshape(1, -1))[0]
    return {str(label): float(v) for label, v in zip(bundle["model"].classes_, p)}


def _alternatives(bundle: dict, task: str, family: str, rows: int, has_gpu: bool, weeks_left, now: dict,
                  top: int = 3) -> list[dict]:
    tries = []
    for other in TASK_FAMILIES[task]:
        if other != family:
            example = FAMILY_EXAMPLE[other].get(task) or FAMILY_EXAMPLE[other]["*"]
            tries.append((f"use {example}", features(task, other, rows, has_gpu, weeks_left)))
    if not has_gpu and family == "deep":
        tries.append(("train on a GPU, for example Google Colab", features(task, family, rows, True, weeks_left)))
    out = []
    for change, x in tries:
        p = _predict(bundle, x)
        label = max(p, key=p.get)
        gain = p["feasible"] - now["feasible"]
        if gain >= MIN_GAIN and RANK[label] <= RANK[max(now, key=now.get)]:
            out.append({"change": change, "label": label, "feasible_probability": round(p["feasible"], 2)})
    out.sort(key=lambda a: -a["feasible_probability"])
    return out[:top]


# ------------------------------------------------------------------- judge
@lru_cache(maxsize=512)
def _judge_cached(bundle_id: int, task: str, algorithm: str, family: str, rows: int, has_gpu: bool,
                  weeks_left: Optional[int]) -> dict:
    bundle = _BUNDLES[bundle_id]
    x = np.array(features(task, family, rows, has_gpu, weeks_left))
    probabilities = _predict(bundle, x)
    label = max(probabilities, key=probabilities.get)
    inputs = {"task": task, "algorithm": algorithm, "family": family, "rows": rows, "has_gpu": has_gpu,
              "weeks_left": weeks_left}
    return {
        "label": label,
        "probability": round(probabilities[label], 2),
        "probabilities": {k: round(v, 2) for k, v in sorted(probabilities.items(), key=lambda kv: RANK[kv[0]])},
        "inputs": inputs,
        "reasons": _reasons(bundle, x, inputs, label),
        "alternatives": [] if label == "feasible" else _alternatives(bundle, task, family, rows, has_gpu,
                                                                     weeks_left, probabilities),
        "note": "Trained on simulated project outcomes; the probabilities are the model's own, not calibrated.",
    }


_BUNDLES: dict[int, dict] = {}


def judge(bundle: dict, task: str, algorithm: str, rows: int, has_gpu: bool = False,
          weeks_left: Optional[float] = None) -> Optional[dict]:
    """The verdict for one plan, or None if task or algorithm is not one the model knows."""
    name, family = _family_of(algorithm)
    task_name = _task_of(task, family)
    if family is None or task_name is None or family not in TASK_FAMILIES[task_name] or not rows:
        return None
    _BUNDLES[id(bundle)] = bundle
    weeks = None if weeks_left is None else max(0, int(weeks_left))      # whole weeks: stable from day to day
    return _judge_cached(id(bundle), task_name, name, family, int(rows), bool(has_gpu), weeks)


def blueprint_inputs(state, events=(), today=None) -> Optional[dict]:
    """task, algorithm, rows, has_gpu and weeks_left from a blueprint, or None if the plan is not
    far enough yet (algorithm and dataset size are needed; the task can come from the algorithm)."""
    def value(name):
        f = state.fields[name]
        return f.value if f.status in WORKING else None

    algorithm, rows = value(FieldName.ALGORITHM), rows_of(value(FieldName.DATASET_SIZE))
    if not algorithm or not rows:
        return None
    compute = set(canonical_names("resource", value(FieldName.RESOURCES)))
    weeks = None
    if state.fields[FieldName.TIMELINE].status in WORKING:
        t = current_timeline(state, events, today)
        if t and t.get("days_left") is not None:
            weeks = max(0.0, t["days_left"] / 7)
    return {"task": value(FieldName.TASK_TYPE) or "", "algorithm": algorithm, "rows": rows,
            "has_gpu": bool(compute & COMPUTE), "weeks_left": weeks}


def judge_blueprint(bundle: Optional[dict], state, events=(), today=None) -> Optional[dict]:
    if bundle is None:
        return None
    inputs = blueprint_inputs(state, events, today)
    if inputs is None:
        return None
    try:
        return judge(bundle, **inputs)
    except Exception:
        log.warning("Component 1 could not run the feasibility model", exc_info=True)
        return None


_default: Optional[dict] = None
_default_tried = False


def default_bundle() -> Optional[dict]:
    """The saved model, loaded once. None if it has not been trained (then the check is skipped)."""
    global _default, _default_tried
    if not _default_tried:
        _default_tried = True
        try:
            if MODEL_PATH.exists():
                _default = load_model(MODEL_PATH)
            else:
                log.warning("Feasibility model not found; run: python -m component1.feasibility.train")
        except Exception:
            log.warning("Component 1 could not load the feasibility model", exc_info=True)
    return _default
