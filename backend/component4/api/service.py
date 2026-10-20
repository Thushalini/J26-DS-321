"""Builds Component 4's full answer for one project: probability, explanation, forecast and recommendations.

Kept apart from the web code so it can be tested and reused without a server.
"""
from datetime import datetime, timezone

import joblib
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


def _factor(item: dict) -> dict:
    name = item["feature"]
    label = config.FEATURE_LABELS.get(name, name)
    value = round(item["value"], 3)
    return {"id": name, "label": label, "value": value, "contribution": item["contribution_points"],
            "detail": f"{label}: {value} {config.KIND_TEXT[get(name).kind]}."}


def predict(features: dict, recurring_causes: list[str] | None = None, source: str = "live",
            with_forecast: bool = True, bundle: dict | None = None) -> dict:
    bundle = bundle or load_bundle()
    model = bundle["model"]
    row, imputed = _row(features, bundle["medians"])
    frame = row.to_frame().T

    explanation = local_explanation(shap_values(model, bundle["background"], frame), 0)
    recurring = [{"causes": [cause], "issues": 1} for cause in recurring_causes or []]
    suggestions = recommend(model, row, explanation["factors"], recurring)
    cone = forecast(model, row, settings={"simulations": config.API["forecast_simulations"]}) if with_forecast else []

    return {
        "source": source,
        "model_version": bundle.get("version"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "probability": round(explanation["probability"] * config.EXPLAINABILITY["probability_to_points"], 1),
        "base_probability": round(explanation["base_probability"] * config.EXPLAINABILITY["probability_to_points"], 1),
        "factors": [_factor(item) for item in explanation["factors"]],
        "forecast": cone,
        "recommendations": [
            {"id": s["id"], "title": s["title"], "reason": s["reason"],
             "area": config.STAGE_LABELS.get(s["stage"], "Whole team"), "stage": s["stage"],
             "expected_gain": s["expected_gain_points"]}
            for s in suggestions
        ],
        "readiness": None,
        "history": [],
        "imputed_features": imputed,
    }


def demo_project(bundle: dict | None = None) -> dict:
    """A held-out synthetic project, so the Predict screen can show real model output before teams have data."""
    bundle = bundle or load_bundle()
    return bundle["demo_project"]
