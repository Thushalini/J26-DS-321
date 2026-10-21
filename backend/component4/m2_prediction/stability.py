"""Seed stability: regenerate the data and re-split with several seeds, keep the tuned settings fixed,
and report how much each model's test results move. Small spread = results don't hinge on one lucky split."""
import numpy as np

from ..common import config
from ..synthetic import generator
from .train import train_and_evaluate


def seed_stability(tuned: dict) -> dict:
    settings = config.SEED_STABILITY
    runs = {}
    for seed in settings["seeds"]:
        report, _ = train_and_evaluate(generator.generate(seed=seed), seed, tuned=tuned, search=False)
        runs[seed] = report["test_metrics"]

    summary = {}
    for model in next(iter(runs.values())):
        summary[model] = {
            metric: {
                "mean": round(float(np.mean([runs[s][model][metric] for s in runs])), 4),
                "std": round(float(np.std([runs[s][model][metric] for s in runs])), 4),
            }
            for metric in settings["metrics"]
        }
    return {"seeds": settings["seeds"], "summary": summary}
