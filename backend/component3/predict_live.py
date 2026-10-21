"""Live prototype: score ONE experiment run and return the result as JSON.

Run from the REPOSITORY ROOT (the folder that contains `backend/`):

    python -m backend.component3.predict_live --run backend/component3/data/samples/sample_run_overfitting.json
    python -m backend.component3.predict_live --demo run_00501        # a run from the dataset
    python -m backend.component3.predict_live --run my_run.json --log  # also append to the issue history

Or from Python:

    from backend.component3.predict_live import predict_run
    result = predict_run(run_dict)

INPUT (a dict / JSON file):
{
  "run_id": "team7_exp12",
  "epochs": [   # one entry per epoch, at least 3
    {"epoch": 1, "train_accuracy": 0.41, "val_accuracy": 0.39, "loss": 1.9,
     "gpu_utilization_pct": 71.0, "commits_that_day": 2}, ...
  ],
  "context": {
    "drift_severity_score": 0.35,          # from Component 2 (0 = on plan, 1 = far off plan)
    "hyperparam_tuning_logged": 1,         # 1 if tuning was logged, else 0
    "commit_gap_days": 2,                  # days since the last commit
    "notebook_edit_freq_per_day": 3.5      # notebook edits per day
  }
}

OUTPUT (JSON): anomaly_score, root_cause_probabilities, severity_score, severity_level,
title, evidence, recommended_cause, second_cause, recommendation, processing_seconds, ...
NOTE: field names are a first version; rename them to match Component 4 / the frontend once agreed.
"""
import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from .common import config
from .common.features import build_run_features
from .stage1_anomaly import anomaly_model
from .stage2_root_cause import root_cause_model as rcm
from .stage3_severity import severity

SCHEMA_VERSION = "0.1-draft"
MIN_EPOCHS = 3          # the shortest run seen in training had 3 epochs
CONTEXT_FIELDS = ["drift_severity_score", "hyperparam_tuning_logged",
                  "commit_gap_days", "notebook_edit_freq_per_day"]
ISSUE_LOG = config.RESULTS_DIR / "issue_history.jsonl"

CAUSE_NAMES = {
    "overfitting": "Overfitting",
    "stalled": "Stalled experiment",
    "data_leakage": "Possible data leakage",
    "resource_bottleneck": "Resource bottleneck",
    "premature_evaluation": "Premature evaluation",
}


def _title_and_evidence(level, top, run_id, df, ctx):
    """Short alert title and a one-line reason built from the run's own numbers."""
    if top is None:
        return "No issue detected", None
    title = f"{CAUSE_NAMES[top]} suspected in {run_id}"
    tr, va = float(df["train_accuracy"].iloc[-1]), float(df["val_accuracy"].iloc[-1])
    loss = df["loss"].to_numpy()
    gpu = float(df["gpu_utilization_pct"].mean())
    if top == "overfitting":
        ev = f"Training accuracy {tr:.0%} vs validation accuracy {va:.0%} (gap {tr - va:.0%})."
    elif top == "stalled":
        k = min(5, len(loss) - 1)
        change = (loss[-k - 1] - loss[-1]) / max(abs(loss[-k - 1]), 1e-9)
        ev = f"Loss improved by only {change:.1%} over the last {k} epochs."
    elif top == "data_leakage":
        ev = f"Validation accuracy of {va:.0%} is unusually high."
    elif top == "resource_bottleneck":
        ev = f"GPU averaged {gpu:.0f}% while validation accuracy reached only {va:.0%}."
    else:
        tuned = "was" if ctx["hyperparam_tuning_logged"] >= 0.5 else "was not"
        ev = f"Evaluated after only {len(df)} epochs; hyperparameter tuning {tuned} logged."
    return title, ev


_models = {}


def _load_models():
    """Load the saved models once and keep them in memory."""
    if not _models:
        _models["anomaly"] = anomaly_model.load()
        _models["root_cause"] = rcm.load()
    return _models["anomaly"], _models["root_cause"]


def _validate(run: dict):
    """Check the input and return (epoch table sorted by epoch, context dict)."""
    if not isinstance(run, dict):
        raise ValueError("Input must be a dict / JSON object.")
    epochs = run.get("epochs")
    if not isinstance(epochs, list) or len(epochs) < MIN_EPOCHS:
        raise ValueError(f"'epochs' must be a list with at least {MIN_EPOCHS} entries.")
    df = pd.DataFrame(epochs)
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)
    missing = [c for c in config.SIGNALS if c not in df.columns]
    if missing:
        raise ValueError(f"Each epoch needs these fields too: {missing}")
    cols = ["epoch"] + config.SIGNALS
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    if df[cols].isna().any().any():
        raise ValueError("Epoch values must be numbers with no empty values.")
    ctx = run.get("context") or {}
    missing = [c for c in CONTEXT_FIELDS if c not in ctx]
    if missing:
        raise ValueError(f"'context' is missing: {missing}")
    try:
        ctx = {c: float(ctx[c]) for c in CONTEXT_FIELDS}
    except (TypeError, ValueError):
        raise ValueError("'context' values must be numbers.")
    return df.sort_values("epoch").reset_index(drop=True), ctx


def predict_run(run: dict, log_path=None) -> dict:
    """Score one run. Returns a JSON-ready dict. If `log_path` is given, the result
    is also appended there as one line (the issue / root-cause history, FR6)."""
    t0 = time.perf_counter()
    a_bundle, rc_bundle = _load_models()
    df, ctx = _validate(run)
    run_id = str(run.get("run_id", "unnamed_run"))

    # Stage 1: anomaly score from the 5 signals
    feats = build_run_features(df.assign(run_id=run_id))
    anomaly = float(anomaly_model.score(a_bundle, feats)[0])

    # Stage 2: root-cause probabilities (summary numbers are worked out from the epochs)
    final_train = float(df["train_accuracy"].iloc[-1])
    final_val = float(df["val_accuracy"].iloc[-1])
    row = pd.DataFrame([{
        "anomaly_score": anomaly,
        "final_train_accuracy": final_train,
        "final_val_accuracy": final_val,
        "train_val_gap": final_train - final_val,
        "n_epochs_run": len(df),
        "avg_gpu_utilization_pct": float(df["gpu_utilization_pct"].mean()),
        **ctx,
    }])
    probs = rcm.predict_probs(rc_bundle, row)[0]

    # Stage 3: severity, level, advice
    sev = float(severity.severity_score(anomaly, probs))
    level = severity.severity_level(sev)
    top, second, advice = severity.recommend(level, probs)
    title, evidence = _title_and_evidence(level, top, run_id, df, ctx)

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_epochs": len(df),
        "anomaly_score": round(anomaly, 4),
        "root_cause_probabilities": {c: round(float(p), 4)
                                     for c, p in zip(config.CAUSES, probs)},
        "severity_score": round(sev, 4),
        "severity_level": level,
        "title": title,
        "evidence": evidence,
        "recommended_cause": top,
        "second_cause": second,
        "recommendation": advice,
        "drift_severity_score_used": ctx["drift_severity_score"],
        "drift_source": str((run.get("context") or {}).get("drift_source", "provided")),
        "assumed": [str(a) for a in run.get("assumed", [])],         # values that were not measured
        "processing_seconds": round(time.perf_counter() - t0, 3),
    }
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    return result


def run_from_dataset(run_id, traj=None, runs=None) -> dict:
    """Build an input dict for a run that is in the dataset (used for the demo and tests)."""
    traj = pd.read_csv(config.TRAJECTORIES_CSV) if traj is None else traj
    runs = pd.read_csv(config.RUN_LABELS_CSV) if runs is None else runs
    g = traj[traj["run_id"] == run_id].sort_values("epoch")
    if g.empty:
        raise ValueError(f"{run_id} not found in the dataset.")
    r = runs[runs["run_id"] == run_id].iloc[0]
    return {"run_id": run_id,
            "epochs": g[["epoch"] + config.SIGNALS].to_dict("records"),
            "context": {c: float(r[c]) for c in CONTEXT_FIELDS}}


def main():
    p = argparse.ArgumentParser(description="Score one experiment run (Component 3).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="path to a run JSON file")
    src.add_argument("--demo", metavar="RUN_ID", help="use a run from the dataset, e.g. run_00501")
    p.add_argument("--log", action="store_true",
                   help=f"also append the result to {ISSUE_LOG.name} in results/")
    args = p.parse_args()
    if args.run:
        with open(args.run, encoding="utf-8") as f:
            run = json.load(f)
    else:
        run = run_from_dataset(args.demo)
    result = predict_run(run, log_path=ISSUE_LOG if args.log else None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
