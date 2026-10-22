"""Simulated project plans for the feasibility classifier, labelled by a simulated OUTCOME.

Why not label plans with rules ("an LSTM on under 1,000 rows is infeasible")? A model trained on
those labels just learns the rules back and scores near 100%, which proves nothing. Here each plan
gets a hidden simulated project: how many hours modelling would really take (training cost x data
size x tuning rounds, plus setup), how many hours the team really has before the deadline, and
whether the data is big enough for that kind of model. Noise stands in for team skill and data
quality, and the true time left is hidden when the team never said it. The model only sees the
plan, so it has to learn a noisy, non-linear relation and cannot be perfect.

Every assumed number is in ASSUMPTIONS below. They are rough, stated on purpose, and should be
cited or adjusted in the report:
  - deep models need far more rows than tree models on tabular data, and trees usually win on
    small and medium tabular data (Grinsztajn, Oyallon and Varoquaux, NeurIPS 2022);
  - linear models need about 10 events per variable (Peduzzi et al., J Clin Epidemiol 1996),
    so a few hundred rows at least;
  - ARIMA-type forecasting needs roughly 50-100 observations (Box and Jenkins; Hyndman and
    Athanasopoulos, Forecasting: Principles and Practice);
  - kernel and distance methods (SVM, KNN) scale worse than linearly with rows;
  - GPUs speed up deep learning training many times over; ~10x is a conservative middle value.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

LABELS = ["feasible", "risky", "infeasible"]
TASKS = ["classification", "regression", "forecasting", "clustering"]
FAMILIES = ["linear", "tree", "distance", "deep", "time_series", "clustering"]

# Which algorithm (standard names from extraction/rules.py) belongs to which family.
ALGORITHM_FAMILY = {
    "Logistic Regression": "linear", "Linear Regression": "linear", "Ridge Regression": "linear",
    "Lasso Regression": "linear", "Naive Bayes": "linear",
    "XGBoost": "tree", "LightGBM": "tree", "CatBoost": "tree", "Random Forest": "tree", "Decision Tree": "tree",
    "Gradient Boosting": "tree", "AdaBoost": "tree",
    "SVM": "distance", "KNN": "distance",
    "Neural Network": "deep", "CNN": "deep", "RNN": "deep", "LSTM": "deep", "GRU": "deep", "Transformer": "deep",
    "BERT": "deep",
    "ARIMA": "time_series", "SARIMA": "time_series", "Prophet": "time_series",
    "K-Means": "clustering", "DBSCAN": "clustering",
}
# Families a team could sensibly use for each task (mismatches are the rule checks' job).
TASK_FAMILIES = {
    "classification": ["linear", "tree", "distance", "deep"],
    "regression": ["linear", "tree", "distance", "deep"],
    "forecasting": ["time_series", "tree", "deep", "linear"],
    "clustering": ["clustering"],
}

ASSUMPTIONS = {
    # hours for ONE training run on 100,000 rows on an ordinary laptop CPU
    "cost_hours_per_100k_rows": {"linear": 0.02, "tree": 0.15, "distance": 0.25, "deep": 1.5,
                                 "time_series": 0.05, "clustering": 0.08},
    "cost_scaling_power": {"linear": 1.0, "tree": 1.0, "distance": 1.6, "deep": 1.0, "time_series": 1.0,
                           "clustering": 1.1},
    "gpu_speedup_deep": 10.0,
    "tuning_runs": {"linear": 5, "tree": 25, "distance": 12, "deep": 30, "time_series": 12, "clustering": 8},
    "setup_hours": {"linear": 6, "tree": 8, "distance": 8, "deep": 30, "time_series": 10, "clustering": 6},
    "min_rows": {"linear": 200, "tree": 500, "distance": 200, "deep": 10_000, "time_series": 100, "clustering": 100},
    # a typical 4-person student team: 8 hours a week each, about a third of it on modelling
    "modelling_hours_per_week": 4 * 8 * 0.35,
    # outcome thresholds
    "infeasible_time_ratio": 1.1,      # needs clearly more modelling hours than the team has
    "risky_time_ratio": 0.45,          # uses more than 45% of them: little room for mistakes
    "infeasible_data_ratio": 0.5,      # under half the rows that kind of model needs
    "risky_data_ratio": 2.5,           # under 2.5x: works, but results will be unstable
    # noise
    "team_skill_sigma": 0.5,           # lognormal spread of hours needed
    "available_time_sigma": 0.3,       # lognormal spread of hours available
    "data_quality_sigma": 0.3,         # lognormal spread of usable rows
    "label_noise": 0.05,
    # what the team says
    "weeks_range": (1, 20),
    "rows_log10_range": (2.0, 7.0),    # 100 to 10 million rows
    "timeline_known_rate": 0.8,
    "weeks_when_unknown": 12,          # what the model is told when nobody said the timeline
    "gpu_rate": 0.4,
}

FEATURE_NAMES = (
    [f"task_{t}" for t in TASKS]
    + [f"family_{f}" for f in FAMILIES]
    + ["log10_rows", "has_gpu", "weeks_left", "timeline_known"]
)
MAX_WEEKS = 26


def features(task: str, family: str, rows: int, has_gpu: bool, weeks_left: Optional[float]) -> list[float]:
    """The numbers the model sees for one plan, in FEATURE_NAMES order. weeks_left None = never said."""
    known = weeks_left is not None
    weeks = ASSUMPTIONS["weeks_when_unknown"] if not known else min(max(float(weeks_left), 0.0), MAX_WEEKS)
    return (
        [1.0 if task == t else 0.0 for t in TASKS]
        + [1.0 if family == f else 0.0 for f in FAMILIES]
        + [math.log10(max(rows, 1)), 1.0 if has_gpu else 0.0, weeks, 1.0 if known else 0.0]
    )


def outcome(rng, family: str, rows: int, has_gpu: bool, true_weeks: float, a: dict = ASSUMPTIONS) -> tuple[str, dict]:
    """The hidden simulated project: (label, details). The model never sees the details."""
    scale = (rows / 100_000) ** a["cost_scaling_power"][family]
    one_run = a["cost_hours_per_100k_rows"][family] * scale
    if family == "deep" and has_gpu:
        one_run /= a["gpu_speedup_deep"]
    needed = (a["setup_hours"][family] + one_run * a["tuning_runs"][family]) * rng.lognormal(0, a["team_skill_sigma"])
    available = max(true_weeks, 0.25) * a["modelling_hours_per_week"] * rng.lognormal(0, a["available_time_sigma"])
    time_ratio = needed / available
    data_ratio = rows / a["min_rows"][family] * rng.lognormal(0, a["data_quality_sigma"])

    if time_ratio > a["infeasible_time_ratio"] or data_ratio < a["infeasible_data_ratio"]:
        label = "infeasible"
    elif time_ratio > a["risky_time_ratio"] or data_ratio < a["risky_data_ratio"]:
        label = "risky"
    else:
        label = "feasible"
    return label, {"hours_needed": needed, "hours_available": available, "time_ratio": time_ratio,
                   "data_ratio": data_ratio}


def simulate_plan(rng, a: dict = ASSUMPTIONS):
    task = str(rng.choice(TASKS, p=[0.35, 0.3, 0.25, 0.1]))
    family = str(rng.choice(TASK_FAMILIES[task]))
    rows = int(10 ** rng.uniform(*a["rows_log10_range"]))
    has_gpu = bool(rng.random() < a["gpu_rate"])
    true_weeks = float(rng.uniform(*a["weeks_range"]))
    said = rng.random() < a["timeline_known_rate"]
    label, _ = outcome(rng, family, rows, has_gpu, true_weeks, a)
    return features(task, family, rows, has_gpu, true_weeks if said else None), label


def generate_dataset(n: int, seed: int = 7, a: dict = ASSUMPTIONS):
    """(X, y): n simulated plans and their outcome labels, with a share of labels flipped."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        x, label = simulate_plan(rng, a)
        if rng.random() < a["label_noise"]:
            label = str(rng.choice([l for l in LABELS if l != label]))
        X.append(x)
        y.append(label)
    return np.array(X), np.array(y)


def rule_baseline(X) -> np.ndarray:
    """The fixed-threshold check the proposal compares against: hand-picked limits, no learning."""
    names = {n: i for i, n in enumerate(FEATURE_NAMES)}
    out = []
    for row in X:
        rows = 10 ** row[names["log10_rows"]]
        deep = row[names["family_deep"]] == 1
        if rows < 500:
            out.append("infeasible")
        elif deep and row[names["has_gpu"]] == 0:
            out.append("risky")
        elif deep and row[names["weeks_left"]] < 4:
            out.append("risky")
        elif rows < 2000:
            out.append("risky")
        else:
            out.append("feasible")
    return np.array(out)
