"""Monte Carlo forecast cone.

Starting from a project's current features, each simulation lets the project drift week by week:
scores take a small random step (optionally pushed by a trend, e.g. "reproducibility keeps improving"),
new experiment issues may appear and open HIGH issues may get resolved. The trained model scores every
simulated week, and the spread of those probabilities is the cone: narrow now, wider further out.
Week 0 is the current prediction.
"""
import numpy as np
import pandas as pd

from ..common import config
from ..synthetic.feature_spec import AT_MOST, FEATURES, RANGES_BY_KIND, ZERO_WHEN_ZERO

COUNT_STEPS = {
    "issue_count": "weekly_new_issues",
    "open_high_issues": "weekly_new_high_issues",
}


def _clip(values: np.ndarray, kind: str) -> np.ndarray:
    low, high = RANGES_BY_KIND[kind]
    return np.clip(values, low, high if high is not None else None)


def _step(state: pd.DataFrame, rng: np.random.Generator, settings: dict, trend: dict) -> pd.DataFrame:
    nxt = state.copy()
    for spec in FEATURES:
        values = state[spec.name].to_numpy(dtype=float)
        if spec.kind in settings["weekly_change"]:
            values = values + rng.normal(0.0, settings["weekly_change"][spec.kind], len(values)) + trend.get(spec.name, 0.0)
            nxt[spec.name] = _clip(values, spec.kind)
    nxt["issue_count"] = state["issue_count"] + rng.poisson(settings[COUNT_STEPS["issue_count"]], len(state))
    resolved = rng.binomial(state["open_high_issues"].astype(int), settings["weekly_resolved_high_share"])
    nxt["open_high_issues"] = state["open_high_issues"] - resolved + rng.poisson(settings[COUNT_STEPS["open_high_issues"]], len(state))
    for smaller, larger in AT_MOST:
        nxt[smaller] = np.minimum(nxt[smaller], nxt[larger])
    for dependent, base in ZERO_WHEN_ZERO:
        nxt.loc[nxt[base] == 0, dependent] = 0
    return nxt


def forecast(model, project: pd.Series, trend: dict | None = None, seed: int = config.SEED,
             settings: dict | None = None) -> list[dict]:
    """Returns one point per week (0 = now) with the probability percentiles, in percent (0-100)."""
    settings = {**config.FORECAST, **(settings or {})}
    rng = np.random.default_rng(seed)
    features = [spec.name for spec in FEATURES]
    state = pd.DataFrame([project[features].to_numpy()] * settings["simulations"], columns=features)
    state = state.astype({spec.name: float for spec in FEATURES})

    weeks = [state]
    for _ in range(settings["weeks"]):
        weeks.append(_step(weeks[-1], rng, settings, trend or {}))

    stacked = pd.concat(weeks, ignore_index=True)
    probabilities = model.predict_proba(stacked)[:, 1].reshape(len(weeks), settings["simulations"]) * 100

    return [
        {"week": week, **{f"p{p}": round(float(np.percentile(row, p)), 1) for p in settings["percentiles"]}}
        for week, row in enumerate(probabilities)
    ]


def cone_checks(points: list[dict], current_probability: float) -> list[dict]:
    """Validation level 5 checks that don't need real history: the cone starts at the current prediction,
    its percentiles stay ordered, and it never narrows going forward."""
    keys = [f"p{p}" for p in config.FORECAST["percentiles"]]
    spread = [point[keys[-1]] - point[keys[0]] for point in points]
    return [
        {"check": "starts_at_current_prediction",
         "passed": all(abs(points[0][k] - current_probability * 100) < 0.11 for k in keys)},
        {"check": "percentiles_ordered",
         "passed": all(all(p[a] <= p[b] for a, b in zip(keys, keys[1:])) for p in points)},
        {"check": "cone_widens_over_time",
         "passed": all(later >= earlier - 1.0 for earlier, later in zip(spread, spread[1:])) and spread[-1] > spread[0]},
    ]
