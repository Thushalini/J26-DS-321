"""Metrics shared by the training pipeline and the generalisation test."""
import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, hamming_loss

from ..common import config


def stage2_metrics(test, probs):
    """Macro F1 over the causes that occur in this test set; Hamming loss and Brier over all five."""
    y = test[config.LABELS].values
    pred = (probs >= 0.5).astype(int)
    present = [i for i in range(5) if y[:, i].sum() > 0]
    return {"macro_F1": float(f1_score(y[:, present], pred[:, present], average="macro", zero_division=0)),
            "hamming_loss": float(hamming_loss(y, pred)),
            "brier": float(np.mean([brier_score_loss(y[:, i], probs[:, i]) for i in range(5)]))}


def alert_summary(df):
    """What a team would experience: false alarms, missed failures, and how often the named cause is right.
    `df` needs: level, is_anomaly, recommended_correct."""
    flagged = df["level"] != "LOW"
    healthy = df["is_anomaly"] == 0
    named_ok = int(((~healthy) & flagged & df["recommended_correct"]).sum())
    n_flag = int(flagged.sum())
    return {"runs": len(df), "healthy_flagged": int(flagged[healthy].sum()), "healthy": int(healthy.sum()),
            "failures_flagged": int(flagged[~healthy].sum()), "failures": int((~healthy).sum()),
            "cause_right_when_flagged": named_ok,
            "alert_precision": float(flagged[~healthy].sum() / n_flag) if n_flag else float("nan")}
