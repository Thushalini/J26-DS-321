"""Create the sample alerts for the Alerts screen from REAL pipeline outputs.

It picks 8 runs from the TEST split of the dataset (one for each of the 5 root causes,
one with two causes at once, and two healthy runs), scores each with predict_live.predict_run(),
and saves the results as one JSON file that the website reads.

Run from the REPOSITORY ROOT (models must exist: python -m backend.component3.train_pipeline):

    python -m backend.component3.scripts.generate_alerts
    python -m backend.component3.scripts.generate_alerts --out frontend/lib/component3-alerts.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ..common import config
from ..predict_live import predict_run, run_from_dataset

REPO_ROOT = config.BASE_DIR.parent.parent
DEFAULT_OUT = REPO_ROOT / "frontend" / "lib" / "component3-alerts.json"


def pick_runs(runs: pd.DataFrame, split: pd.DataFrame) -> list:
    """8 test-split runs: first of each failure type, one multi-cause run, two healthy."""
    d = runs.merge(split, on="run_id")
    d = d[d["split"] == "test"].sort_values("run_id")
    picked = []
    for t in ["overfitting", "stalled", "data_leakage", "resource_bottleneck", "premature_evaluation"]:
        picked.append(d[d["run_type"] == t].iloc[0]["run_id"])
    multi = d[(d[config.LABELS].sum(axis=1) > 1) & (~d["run_id"].isin(picked))]
    if len(multi):
        picked.append(multi.iloc[0]["run_id"])
    picked += list(d[d["run_type"] == "healthy"]["run_id"].iloc[:2])
    return picked


def main():
    p = argparse.ArgumentParser(description="Generate sample alerts from real pipeline outputs.")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the JSON file")
    args = p.parse_args()

    traj = pd.read_csv(config.TRAJECTORIES_CSV)
    runs = pd.read_csv(config.RUN_LABELS_CSV)
    split = pd.read_csv(config.SPLITS_CSV)

    alerts = []
    for i, run_id in enumerate(pick_runs(runs, split), start=1):
        res = predict_run(run_from_dataset(run_id, traj, runs))
        res["id"] = f"alert-{i}"
        res["status"] = "open" if res["severity_level"] != "LOW" else "healthy"
        alerts.append(res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": "Component 3 pipeline outputs on sample runs",
                               "alerts": alerts}, indent=2), encoding="utf-8")
    print(f"Saved {len(alerts)} alerts to {out}")
    for a in alerts:
        print(f"  {a['id']}  {a['severity_level']:<6} {a['title']}")


if __name__ == "__main__":
    main()
