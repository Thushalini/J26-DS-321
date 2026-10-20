"""The real training runs (made by scripts/make_real_runs.py) and their train / validation / test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def available():
    return config.REAL_TRAJECTORIES_CSV.exists() and config.REAL_RUN_LABELS_CSV.exists()


def load_real_runs():
    """(epoch table, one-row-per-run labels), or (None, None) when the real runs are not there."""
    if not available():
        return None, None
    return pd.read_csv(config.REAL_TRAJECTORIES_CSV), pd.read_csv(config.REAL_RUN_LABELS_CSV)


def load_or_make_real_splits(labels):
    """70 / 15 / 15 by run, with every run type in proportion. Saved so every result uses the same split."""
    if config.REAL_SPLITS_CSV.exists():
        return pd.read_csv(config.REAL_SPLITS_CSV)
    train_ids, temp_ids = train_test_split(labels["run_id"], test_size=0.30, random_state=config.SEED,
                                           stratify=labels["run_type"])
    temp = labels[labels["run_id"].isin(temp_ids)]
    val_ids, test_ids = train_test_split(temp["run_id"], test_size=0.50, random_state=config.SEED,
                                         stratify=temp["run_type"])
    split = pd.concat([pd.DataFrame({"run_id": train_ids, "split": "train"}),
                       pd.DataFrame({"run_id": val_ids, "split": "val"}),
                       pd.DataFrame({"run_id": test_ids, "split": "test"})])
    split.to_csv(config.REAL_SPLITS_CSV, index=False)
    return split
