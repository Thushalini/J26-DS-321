"""Shared settings for Component 3 (paths, names, random seed)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # the component3 folder
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

TRAJECTORIES_CSV = DATA_DIR / "experiment_trajectories.csv"
RUN_LABELS_CSV = DATA_DIR / "run_level_labels.csv"
SPLITS_CSV = DATA_DIR / "splits.csv"

# Real training runs (made by scripts/make_real_runs.py)
REAL_DIR = DATA_DIR / "real_runs"
REAL_TRAJECTORIES_CSV = REAL_DIR / "real_trajectories.csv"
REAL_RUN_LABELS_CSV = REAL_DIR / "real_run_labels.csv"
REAL_SPLITS_CSV = REAL_DIR / "real_splits.csv"

ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.joblib"
ROOT_CAUSE_MODEL_PATH = MODELS_DIR / "root_cause_models.joblib"

SEED = 42

# The 5 raw signals recorded every epoch
SIGNALS = ["train_accuracy", "val_accuracy", "loss",
           "gpu_utilization_pct", "commits_that_day"]

# The 5 root causes (one yes/no label column each: cause_<name>)
CAUSES = ["overfitting", "stalled", "data_leakage",
          "resource_bottleneck", "premature_evaluation"]
LABELS = [f"cause_{c}" for c in CAUSES]
PROB_COLS = [f"prob_{c}" for c in CAUSES]

# Inputs of the root-cause classifier (anomaly_score comes from Stage 1)
ROOT_CAUSE_FEATURES = [
    "anomaly_score",
    "final_train_accuracy", "final_val_accuracy", "train_val_gap",
    "n_epochs_run", "avg_gpu_utilization_pct",
    "hyperparam_tuning_logged", "drift_severity_score",
    "commit_gap_days", "notebook_edit_freq_per_day",
]
