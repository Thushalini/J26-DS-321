"""Make a sheet of invented project plans for a person to label feasible / risky / infeasible.

Run from the backend folder:
    python -m component1.feasibility.expert_sheet --out component1/data/feasibility_expert.csv --n 40

Give the sheet to a supervisor or an experienced data scientist. They fill in the "label" column
(feasible, risky or infeasible) for each plan, using their own judgement, WITHOUT seeing what the
model says. The sheet deliberately does not contain the model's answers. Then:
    python -m component1.feasibility.train --expert component1/data/feasibility_expert.csv

That score, reported separately from the simulated one, is the honest check of the model.
The plans are spread over tasks, model families, sizes, GPU and time, so every kind of plan appears.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .predict import FAMILY_EXAMPLE
from .simulate import ASSUMPTIONS, TASK_FAMILIES, TASKS

COLUMNS = ["plan_id", "task", "algorithm", "rows", "gpu", "weeks_left", "label", "notes"]
INSTRUCTIONS = ("Label each plan feasible, risky or infeasible: could a 4-person student team, "
                "spending about a day a week each, finish the modelling in the time left?")


def make_plans(n: int = 40, seed: int = 11) -> list[dict]:
    rng = np.random.default_rng(seed)
    pairs = [(t, f) for t in TASKS for f in TASK_FAMILIES[t]]
    plans = []
    for i in range(n):
        task, family = pairs[i % len(pairs)]                        # every task/family pair appears
        algorithm = FAMILY_EXAMPLE[family].get(task) or FAMILY_EXAMPLE[family]["*"]
        rows = int(round(10 ** rng.uniform(*ASSUMPTIONS["rows_log10_range"]), -1)) or 100
        plans.append({
            "plan_id": f"P{i + 1:02d}",
            "task": task,
            "algorithm": algorithm,
            "rows": rows,
            "gpu": "yes" if rng.random() < 0.4 else "no",
            "weeks_left": int(rng.integers(1, 17)),
            "label": "",
            "notes": "",
        })
    order = rng.permutation(len(plans))                             # not grouped by kind
    return [plans[i] for i in order]


def write_sheet(path: Path, plans: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(plans)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("component1/data/feasibility_expert.csv"))
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args(argv)
    write_sheet(args.out, make_plans(args.n))
    print(f"Wrote {args.n} plans to {args.out}")
    print(INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
