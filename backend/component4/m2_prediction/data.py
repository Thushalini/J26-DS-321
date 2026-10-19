"""Turns the project dataset into model inputs and a stratified train / validation / test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from ..common import config
from ..synthetic.feature_spec import FEATURES


def prediction_features() -> list[str]:
    allowed = [spec.name for spec in FEATURES if spec.source not in config.FORBIDDEN_FEATURE_SOURCES]
    if len(allowed) != len(FEATURES):
        raise ValueError("A feature from a forbidden source (academic marks) is in the prediction schema.")
    return allowed


def success_label(frame: pd.DataFrame) -> pd.Series:
    return frame[config.OUTCOME_COLUMN].isin(config.POSITIVE_OUTCOMES).astype(int)


def split(frame: pd.DataFrame, seed: int = config.SEED) -> dict:
    shares = config.SPLIT
    label = success_label(frame)
    train, rest = train_test_split(frame, train_size=shares["train"], stratify=label, random_state=seed)
    val_share = shares["val"] / (shares["val"] + shares["test"])
    val, test = train_test_split(rest, train_size=val_share, stratify=success_label(rest), random_state=seed)

    features = prediction_features()
    return {name: (part[features], success_label(part)) for name, part in
            (("train", train), ("val", val), ("test", test))}
