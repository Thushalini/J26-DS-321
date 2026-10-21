"""Builds Component 4's full answer for one project: probability, explanation, forecast and recommendations.

Kept apart from the web code so it can be tested and reused without a server.
"""
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from ..common import config
from ..m3_explainability.shap_explainer import local_explanation, shap_values
from ..m5_forecast.cone import forecast
from ..m6_improvement.recommendations import recommend
from ..synthetic.feature_spec import feature_names, get

_bundle = {}


class ModelNotReady(RuntimeError):
    pass


def load_bundle(path=None) -> dict:
    path = path or config.MODEL_PATH
    if _bundle.get("path") != path:
        if not path.exists():
            raise ModelNotReady("No trained model yet. Run: python -m component4.train_pipeline")
        bundle = joblib.load(path)
        if "background" not in bundle or "medians" not in bundle:
            raise ModelNotReady("The saved model is from an older version. Run: python -m component4.train_pipeline")
        _bundle.clear()
        _bundle.update(bundle, path=path)
    return _bundle


def _row(features: dict, medians: dict) -> tuple[pd.Series, list[str]]:
    imputed = [name for name in feature_names() if features.get(name) is None]
    values = {name: float(medians[name] if features.get(name) is None else features[name]) for name in feature_names()}
    return pd.Series(values), imputed


def _factor(item: dict, measured: bool) -> dict:
    name = item["feature"]
    label = config.FEATURE_LABELS.get(name, name)
    value = round(item["value"], 3)
    return {"id": name, "label": label, "value": value, "contribution": item["contribution_points"],
            "measured": measured, "detail": f"{label}: {value} {config.KIND_TEXT[get(name).kind]}."}


def probability_range(model, row: pd.Series, imputed: list[str], sample: pd.DataFrame) -> dict:
    """Monte Carlo over the signals not measured yet: fill them with real combinations from training projects
    and see how far the probability moves. With every signal measured the range is the single value."""
    settings = config.API
    points = config.EXPLAINABILITY["probability_to_points"]
    rows = pd.DataFrame([row] * settings["range_simulations"]).reset_index(drop=True)
    if imputed:
        picks = sample.sample(settings["range_simulations"], replace=True, random_state=config.SEED)
        rows[imputed] = picks[imputed].to_numpy()
    probabilities = model.predict_proba(rows[feature_names()])[:, 1] * points
    low, high = np.quantile(probabilities, settings["range_quantiles"])
    return {"low": round(float(low), 1), "high": round(float(high), 1)}


def quick(features: dict, bundle: dict | None = None) -> dict:
    """Probability only, without explanation or forecast: cheap enough for a supervisor's whole group list."""
    bundle = bundle or load_bundle()
    row, imputed = _row(features, bundle["medians"])
    probability = float(bundle["model"].predict_proba(row.to_frame().T[feature_names()])[:, 1][0])
    return {"probability": round(probability * config.EXPLAINABILITY["probability_to_points"], 1),
            "imputed_features": imputed, "evaluated_at": now(), "model_version": bundle.get("version")}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def predict(features: dict, recurring_causes: list[str] | None = None, source: str = "live",
            with_forecast: bool = True, bundle: dict | None = None, readiness: dict | None = None,
            recurring_counts: dict[str, int] | None = None) -> dict:
    """Recommendations only use measured signals: a typical value filled in for a missing one is not evidence."""
    bundle = bundle or load_bundle()
    model = bundle["model"]
    row, imputed = _row(features, bundle["medians"])
    frame = row.to_frame().T

    explanation = local_explanation(shap_values(model, bundle["background"], frame), 0)
    recurring = [{"causes": [cause], "issues": (recurring_counts or {}).get(cause, 1)} for cause in recurring_causes or []]
    measured = [item for item in explanation["factors"] if item["feature"] not in imputed]
    suggestions = recommend(model, row, measured, recurring)
    cone = forecast(model, row, settings={"simulations": config.API["forecast_simulations"]}) if with_forecast else []
    probability = round(explanation["probability"] * config.EXPLAINABILITY["probability_to_points"], 1)
    spread = probability_range(model, row, imputed, bundle.get("fill_sample", bundle["background"]))
    spread = {"low": min(spread["low"], probability), "high": max(spread["high"], probability)}

    return {
        "source": source,
        "model_version": bundle.get("version"),
        "evaluated_at": now(),
        "probability": probability,
        "base_probability": round(explanation["base_probability"] * config.EXPLAINABILITY["probability_to_points"], 1),
        "probability_range": spread,
        "factors": [_factor(item, item["feature"] not in imputed) for item in explanation["factors"]],
        "forecast": cone,
        "recommendations": [
            {"id": s["id"], "title": s["title"], "reason": s["reason"],
             "area": config.STAGE_LABELS.get(s["stage"], "Whole team"), "stage": s["stage"],
             "expected_gain": s["expected_gain_points"]}
            for s in suggestions
        ],
        "readiness": readiness,
        "history": [],
        "imputed_features": imputed,
    }


def demo_project(bundle: dict | None = None) -> dict:
    """A held-out synthetic project, so the Predict screen can show real model output before teams have data."""
    bundle = bundle or load_bundle()
    return bundle["demo_project"]
