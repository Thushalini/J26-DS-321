"""Use the saved decision-intent classifier on one message.

    bundle = load_model()
    label, probability = predict_intent(bundle, "I agree", prev_message="Let's use XGBoost")

The probability is the model's own confidence. It is NOT calibrated.
"""
from __future__ import annotations

from pathlib import Path

import joblib

from .data import Turn

MODEL_PATH = Path(__file__).parent / "models" / "intent_classifier.joblib"


def load_model(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_intent(bundle: dict, message: str, prev_message: str = "") -> tuple[str, float]:
    turn = Turn("", 0, "", message.strip(), prev_message.strip(), "other")
    probabilities = bundle["model"].predict_proba(bundle["features"].transform([turn]))[0]
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    return str(bundle["model"].classes_[best]), float(probabilities[best])