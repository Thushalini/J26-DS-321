"""Use the saved state classifier to judge one field.

    bundle = load_model()
    label, probability = predict_status(bundle, field_features(events, "target_variable"))

The probability is the model's own confidence in that label. It is NOT calibrated,
so do not read "0.82" as "right 82% of the time".
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib

from .features import FEATURE_ORDER

MODEL_PATH = Path(__file__).parent / "models" / "state_classifier.joblib"


def load_model(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_status(bundle: dict, features: Optional[list[float]]) -> tuple[str, float]:
    """Return (status, probability). A field that was never mentioned is "missing"."""
    if features is None:
        return "missing", 1.0
    if list(bundle["features"]) != FEATURE_ORDER:
        raise ValueError(
            f"The saved model uses features {bundle['features']}, but the pipeline builds "
            f"{FEATURE_ORDER}. Retrain with: python -m component1.state.train"
        )
    model = bundle["model"]
    probabilities = model.predict_proba([features])[0]
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    return str(model.classes_[best]), float(probabilities[best])