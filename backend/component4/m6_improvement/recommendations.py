"""Improvement recommendations: what to fix next, and how much it should help.

An action is considered when one of its signals is pulling this project's probability down (a negative
SHAP contribution) or when one of its root causes keeps recurring in the team's issue history.
Its expected gain comes from re-scoring the project with the action's change applied, so every
suggestion is backed by the model rather than a fixed rule of thumb. Academic marks are not used here;
choosing who could help is the separate competency branch.
"""
import numpy as np
import pandas as pd

from ..common import config
from ..synthetic.feature_spec import AT_MOST, RANGES_BY_KIND, ZERO_WHEN_ZERO, get

SETTINGS = config.RECOMMENDATIONS
POINTS = config.EXPLAINABILITY["probability_to_points"]


def apply_changes(project: pd.Series, changes: dict) -> pd.Series:
    after = project.astype(float).copy()
    for name, change in changes.items():
        low, high = RANGES_BY_KIND[get(name).kind]
        value = 0.0 if change == "zero" else after[name] + change
        after[name] = float(np.clip(value, low, high if high is not None else None))
    for smaller, larger in AT_MOST:
        after[smaller] = min(after[smaller], after[larger])
    for dependent, base in ZERO_WHEN_ZERO:
        if after[base] == 0:
            after[dependent] = 0.0
    return after


def _probability(model, project: pd.Series) -> float:
    return float(model.predict_proba(project.to_frame().T.astype(float))[:, 1][0])


def _reasons(action: dict, weak: dict, recurring_causes: dict) -> list[str]:
    reasons = [f"{name.replace('_', ' ')} is lowering the probability by {abs(points):.1f} pts"
               for name, points in weak.items() if name in action["when_weak"]]
    reasons += [f"recurring issue pattern: {cause.replace('_', ' ')} ({count} issues)"
                for cause, count in recurring_causes.items() if cause in action["when_pattern"]]
    return reasons


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] + "."


def recommend(model, project: pd.Series, factors: list[dict], recurring: list[dict] | None = None) -> list[dict]:
    """factors: the project's local SHAP explanation; recurring: recurring_for_team() output."""
    weak = {f["feature"]: f["contribution_points"] for f in factors
            if f["contribution_points"] <= SETTINGS["weak_at_points"]}
    recurring_causes = {}
    for pattern in recurring or []:
        for cause in pattern["causes"]:
            recurring_causes[cause] = recurring_causes.get(cause, 0) + pattern["issues"]

    now = _probability(model, project)
    suggestions = []
    for action_id, action in SETTINGS["actions"].items():
        reasons = _reasons(action, weak, recurring_causes)
        if not reasons:
            continue
        gain = (_probability(model, apply_changes(project, action["changes"])) - now) * POINTS
        if gain < SETTINGS["min_gain_points"]:
            continue
        suggestions.append({
            "id": action_id,
            "title": action["title"],
            "reason": _sentence("; ".join(reasons)),
            "stage": action["stage"],
            "expected_gain_points": round(gain, 1),
        })
    suggestions.sort(key=lambda s: s["expected_gain_points"], reverse=True)
    return suggestions[:SETTINGS["top_n"]]


def re_evaluate(model, before: pd.Series, after: pd.Series) -> dict:
    """Validation level 6: the same project before and after the team acted on a recommendation."""
    p_before, p_after = _probability(model, before), _probability(model, after)
    changed = {name: {"before": float(before[name]), "after": float(after[name])}
               for name in before.index if float(before[name]) != float(after[name])}
    return {"probability_before": round(p_before, 4), "probability_after": round(p_after, 4),
            "change_points": round((p_after - p_before) * POINTS, 1), "changed_features": changed}
