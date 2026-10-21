"""Validation level 4: do the explanations and the model behave the way the domain says they should?

Direction check: across many projects, a feature's SHAP value should rise with the feature when
it helps success and fall when it hurts.
Sensitivity analysis: nudging one feature in its "better" direction, everything else fixed,
should not lower the predicted probability.
"""
import numpy as np
import pandas as pd
import shap

from ..common import config
from ..synthetic.feature_spec import FEATURES, RANGES_BY_KIND


def direction_check(explanation: shap.Explanation) -> list[dict]:
    minimum = config.EXPLAINABILITY["min_direction_correlation"]
    data = pd.DataFrame(explanation.data, columns=explanation.feature_names)
    results = []
    for spec in FEATURES:
        if spec.direction == 0:
            results.append({"feature": spec.name, "expected": 0, "passed": True, "note": "neutral, not checked"})
            continue
        column = explanation.feature_names.index(spec.name)
        rho = pd.Series(explanation.values[:, column]).corr(data[spec.name], method="spearman")
        rho = 0.0 if pd.isna(rho) else float(rho)
        results.append({
            "feature": spec.name,
            "expected": spec.direction,
            "shap_value_correlation": round(rho, 3),
            "passed": rho * spec.direction >= minimum,
        })
    return results


def sensitivity_analysis(model, rows: pd.DataFrame) -> list[dict]:
    settings = config.SENSITIVITY
    baseline = model.predict_proba(rows)[:, 1]
    results = []
    for spec in FEATURES:
        if spec.direction == 0:
            results.append({"feature": spec.name, "step": 0, "passed": True, "note": "neutral, not checked"})
            continue
        low, high = RANGES_BY_KIND[spec.kind]
        nudged = rows.copy()
        moved = nudged[spec.name] + spec.direction * settings["steps"][spec.kind]
        nudged[spec.name] = moved.clip(lower=low, upper=high).astype(rows[spec.name].dtype)
        change = model.predict_proba(nudged)[:, 1] - baseline
        share_ok = float(np.mean(change >= 0))
        results.append({
            "feature": spec.name,
            "step": spec.direction * settings["steps"][spec.kind],
            "mean_change_points": round(float(change.mean() * config.EXPLAINABILITY["probability_to_points"]), 3),
            "share_not_lower": round(share_ok, 3),
            "passed": share_ok >= settings["min_share_in_expected_direction"],
        })
    return results
