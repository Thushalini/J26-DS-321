"""Make a dataset of REAL training runs with known failures (the curves are never simulated).

Every row comes from an actual scikit-learn model trained epoch by epoch. We cause each failure on
purpose, so its label is known. Tasks, models and optimisers are varied on purpose, so the detector
learns what healthy training looks like in general and not only on one kind of task.

  healthy              normal training
  overfitting          small noisy training set, big network, many epochs
  stalled              learning rate far too small, so the loss hardly moves
  data_leakage         the label sneaks into the features
  premature_evaluation stopped after only a few epochs

Not covered: resource_bottleneck (needs a GPU to measure; the simulated runs still cover it).
Not measured here, so filled with VARIED healthy-looking random values: GPU use, commits, commit gap,
notebook edits, drift score. Only the training curves and the number of epochs are real.

Each run records its task and model, so whole families can be held out when testing.

    python -m backend.component3.scripts.make_real_runs [--scale 1.0] [--out DIR]
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import (load_breast_cancer, load_digits, load_iris, load_wine, make_blobs,
                              make_circles, make_classification, make_moons)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ..common import config

warnings.filterwarnings("ignore")
CAUSES = config.CAUSES
PLAN = {"healthy": 400, "overfitting": 200, "stalled": 150, "data_leakage": 130, "premature_evaluation": 120}
TASKS = ["classif", "moons", "circles", "blobs", "digits", "breast_cancer", "wine", "iris"]
MODELS = ["mlp_adam", "mlp_sgd", "sgd_logreg"]


def make_task(task, rng, seed):
    if task == "classif":
        k = int(rng.choice([2, 3, 5, 8]))
        X, y = make_classification(n_samples=int(rng.integers(1000, 3000)), n_features=int(rng.integers(15, 40)),
                                   n_informative=int(rng.integers(6, 12)), n_redundant=3, n_classes=k,
                                   n_clusters_per_class=1, flip_y=float(rng.uniform(0.02, 0.15)),
                                   class_sep=float(rng.uniform(0.6, 1.6)), random_state=seed)
    elif task == "moons":
        X, y = make_moons(n_samples=int(rng.integers(600, 2000)), noise=float(rng.uniform(0.15, 0.4)), random_state=seed)
    elif task == "circles":
        X, y = make_circles(n_samples=int(rng.integers(600, 2000)), noise=float(rng.uniform(0.08, 0.2)), factor=0.5, random_state=seed)
    elif task == "blobs":
        X, y = make_blobs(n_samples=int(rng.integers(600, 2000)), centers=int(rng.integers(3, 7)),
                          cluster_std=float(rng.uniform(1.5, 3.5)), n_features=int(rng.integers(4, 12)), random_state=seed)
    elif task == "digits":
        X, y = load_digits(return_X_y=True)
    elif task == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
    elif task == "wine":
        X, y = load_wine(return_X_y=True)
    else:
        X, y = load_iris(return_X_y=True)
    return StandardScaler().fit_transform(X), y


def make_model(kind, rng, seed, scenario):
    lr = float(rng.choice([0.003, 0.01, 0.03]))
    if scenario == "stalled":
        lr = float(rng.choice([1e-6, 3e-6, 1e-5, 3e-5]))
    if kind == "sgd_logreg":
        return SGDClassifier(loss="log_loss", learning_rate="constant", eta0=lr * 3, alpha=float(rng.choice([1e-5, 1e-4])),
                             random_state=seed), "sgd_logreg"
    hidden = [(16,), (32,), (64,), (32, 16)][int(rng.integers(0, 4))]
    if scenario == "overfitting":
        hidden = [(128, 128), (256, 256), (256,)][int(rng.integers(0, 3))]
    solver = "adam" if kind == "mlp_adam" else "sgd"
    kw = {"momentum": 0.9} if solver == "sgd" else {}
    return MLPClassifier(hidden_layer_sizes=hidden, solver=solver, learning_rate_init=lr * (3 if solver == "sgd" else 1),
                         batch_size=int(rng.choice([16, 32, 64])), random_state=seed, **kw), kind


def train_one(scenario, seed):
    rng = np.random.default_rng(seed)
    task, kind = TASKS[int(rng.integers(0, len(TASKS)))], MODELS[int(rng.integers(0, len(MODELS)))]
    X, y = make_task(task, rng, seed)
    if scenario == "data_leakage":
        X = np.column_stack([X, y + rng.normal(0, float(rng.uniform(0.02, 0.3)), size=len(y))])
    strat = y if np.bincount(y).min() >= 2 else None
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=strat)
    epochs = int(rng.integers(12, 45))
    if scenario == "overfitting":
        n = int(min(rng.integers(40, 160), len(Xtr)))
        Xtr, ytr = Xtr[:n].copy(), ytr[:n].copy()
        flip = rng.random(n) < float(rng.uniform(0.1, 0.3))          # noisy labels get memorised
        ytr[flip] = rng.choice(np.unique(y), flip.sum())
        epochs = int(rng.integers(30, 60))
    elif scenario == "premature_evaluation":
        epochs = int(rng.integers(3, 7))
    model, kind = make_model(kind, rng, seed, scenario)
    classes = np.unique(y)
    rows = []
    for e in range(1, epochs + 1):
        model.partial_fit(Xtr, ytr, classes=classes)
        loss = float(log_loss(ytr, model.predict_proba(Xtr), labels=classes))     # same loss definition for every model
        rows.append((e, float(model.score(Xtr, ytr)), float(model.score(Xva, yva)), loss))
    return pd.DataFrame(rows, columns=["epoch", "train_accuracy", "val_accuracy", "loss"]), rng, task, kind


def manifested(scenario, df):
    """Keep an injected failure only if it really showed up; drop 'healthy' runs that overfit or did not learn."""
    gap = df.train_accuracy.iloc[-1] - df.val_accuracy.iloc[-1]
    loss_drop = (df.loss.iloc[0] - df.loss.iloc[-1]) / max(abs(df.loss.iloc[0]), 1e-9)
    if scenario == "healthy":
        return gap < 0.12 and (loss_drop > 0.10 or df.train_accuracy.iloc[0] > 0.9)
    if scenario == "overfitting":
        return gap > 0.12
    if scenario == "stalled":
        return loss_drop < 0.10
    if scenario == "data_leakage":
        return df.val_accuracy.iloc[-1] > 0.93
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the number of runs (0.1 for a quick test)")
    ap.add_argument("--out", default=str(config.DATA_DIR / "real_runs"))
    ap.add_argument("--fresh", action="store_true", help="start again instead of continuing a saved partial set")
    a = ap.parse_args()
    from pathlib import Path
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tf, lf = out / "real_trajectories.csv", out / "real_run_labels.csv"
    traj_rows, label_rows, next_seed = [], [], 7000
    if lf.exists() and tf.exists() and not a.fresh:            # continue where an earlier (interrupted) run stopped
        old = pd.read_csv(lf)
        label_rows = old.to_dict("records")
        traj_rows = [pd.read_csv(tf)]
        next_seed = int(old["seed"].max()) + 1
        print(f"continuing from {len(old)} saved runs", flush=True)

    def save():
        pd.concat(traj_rows).to_csv(tf, index=False)
        pd.DataFrame(label_rows).to_csv(lf, index=False)

    for scenario, base in PLAN.items():
        target = max(1, int(base * a.scale))
        kept = sum(1 for r in label_rows if r["run_type"] == scenario)
        attempts = dropped = 0
        while kept < target and attempts < target * 6:
            seed, next_seed, attempts = next_seed, next_seed + 1, attempts + 1
            try:
                df, rng, task, kind = train_one(scenario, seed)
            except (ValueError, FloatingPointError):            # a network that blew up (not one of our labelled failures)
                dropped += 1
                continue
            if not manifested(scenario, df):
                dropped += 1
                continue
            run_id = f"real_{seed}"
            gpu = float(rng.uniform(40, 92))
            df["gpu_utilization_pct"] = np.clip(gpu + rng.normal(0, 3, len(df)), 0, 100).round(2)
            df["commits_that_day"] = rng.poisson(float(rng.uniform(1, 4)), len(df))
            df.insert(0, "run_id", run_id)
            traj_rows.append(df)
            tuned = int(rng.random() < (0.2 if scenario == "premature_evaluation" else 0.85))
            label_rows.append({
                "run_id": run_id, "seed": seed, "run_type": scenario, "is_anomaly": int(scenario != "healthy"),
                "final_train_accuracy": round(df.train_accuracy.iloc[-1], 4),
                "final_val_accuracy": round(df.val_accuracy.iloc[-1], 4),
                "train_val_gap": round(df.train_accuracy.iloc[-1] - df.val_accuracy.iloc[-1], 3),
                "n_epochs_run": len(df), "avg_gpu_utilization_pct": round(float(df.gpu_utilization_pct.mean()), 2),
                "hyperparam_tuning_logged": tuned, "drift_severity_score": round(float(rng.uniform(0, 0.5)), 3),
                "commit_gap_days": round(float(rng.uniform(0, 2.5)), 2),
                "notebook_edit_freq_per_day": round(float(rng.uniform(1, 8)), 2),
                "task": task, "model": kind,
                **{f"cause_{c}": int(c == scenario) for c in CAUSES},
            })
            kept += 1
            if kept % 20 == 0:
                save()
        save()
        print(f"{scenario:<21} {kept}/{target} (dropped {dropped} this time where the failure did not show up)", flush=True)
    print(f"Saved {len(label_rows)} real runs to {out}")


if __name__ == "__main__":
    main()
