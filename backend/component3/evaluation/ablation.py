"""Ablation study: rule-based baselines vs trained ML, plus feature ablations."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score

from ..common import config
from ..stage2_root_cause import root_cause_model as rcm
from . import baseline


def _summary(name, y_true, y_pred, is_anomaly, det_score):
    present = [i for i in range(y_true.shape[1]) if y_true[:, i].sum() > 0]     # causes that occur in this test set
    flagged = np.asarray(y_pred).sum(axis=1) > 0                                # the system names at least one cause
    healthy = np.asarray(y_true).sum(axis=1) == 0
    return {"system": name,
            "macro_F1": f1_score(y_true[:, present], y_pred[:, present], average="macro", zero_division=0),
            "hamming_loss": hamming_loss(y_true, y_pred),
            "detection_AUC": roc_auc_score(is_anomaly, det_score),
            "healthy_flagged": float(flagged[healthy].mean()) if healthy.any() else float("nan"),
            "failures_flagged": float(flagged[~healthy].mean()) if (~healthy).any() else float("nan")}


def run_ablation(data, rule_a, rc_bundle):
    """`data`: one row per run with labels, features, split, anomaly_score.
    `rule_a`: Baseline A flags (from baseline.baseline_a). Evaluated on the TEST split."""
    d = data.merge(rule_a, on="run_id")
    train = d[d["split"] == "train"]
    test = d[d["split"] == "test"]
    y = test[config.LABELS].values
    feats = rc_bundle["features"]

    a_pred = test[baseline.RULE_A_COLS].values
    b_pred = baseline.baseline_b(train, test)

    ml_full = rcm.predict_probs(rc_bundle, test)

    def variant(drop):
        f = [x for x in feats if x != drop]
        models = rcm.fit_models(train, f, rc_bundle["params"])
        return rcm.predict_probs_from(models, test, f)

    ml_no_anom = variant("anomaly_score")
    ml_no_drift = variant("drift_severity_score")

    ia = test["is_anomaly"]
    rows = [
        _summary("Baseline A (generic rules)", y, a_pred, ia, a_pred.sum(axis=1)),
        _summary("Baseline B (best single rule per cause, fit on train)", y, b_pred, ia,
                 b_pred.sum(axis=1)),
        _summary("Trained ML (full)", y, (ml_full >= 0.5).astype(int), ia, test["anomaly_score"]),
        _summary("ML without anomaly score", y, (ml_no_anom >= 0.5).astype(int), ia,
                 ml_no_anom.max(axis=1)),
        _summary("ML without drift score", y, (ml_no_drift >= 0.5).astype(int), ia,
                 ml_no_drift.max(axis=1)),
    ]
    table = pd.DataFrame(rows).round(4)

    # bootstrap: how sure are we about the F1 gain?
    rng = np.random.default_rng(config.SEED)
    ml_pred = (ml_full >= 0.5).astype(int)
    gain_a, gain_b = [], []
    for _ in range(1000):
        idx = rng.integers(0, len(test), len(test))
        f_ml = f1_score(y[idx], ml_pred[idx], average="macro", zero_division=0)
        gain_a.append(f_ml - f1_score(y[idx], a_pred[idx], average="macro", zero_division=0))
        gain_b.append(f_ml - f1_score(y[idx], b_pred[idx], average="macro", zero_division=0))
    ci = {"vs_baseline_A": [float(x) for x in np.percentile(gain_a, [2.5, 97.5])],
          "vs_baseline_B": [float(x) for x in np.percentile(gain_b, [2.5, 97.5])]}

    per_cause = pd.DataFrame({
        "cause": config.CAUSES,
        "F1_baseline_A": [f1_score(y[:, i], a_pred[:, i], zero_division=0) for i in range(5)],
        "F1_baseline_B": [f1_score(y[:, i], b_pred[:, i], zero_division=0) for i in range(5)],
        "F1_ML": [f1_score(y[:, i], ml_pred[:, i], zero_division=0) for i in range(5)],
    }).round(3)
    return table, ci, per_cause
