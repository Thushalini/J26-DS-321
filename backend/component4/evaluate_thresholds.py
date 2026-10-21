"""Threshold sensitivity for the evaluation: how much results move when each provisional threshold is nudged.

Synthetic part (always): prediction quality and today's range width by number of measured signals (the
evidence for min_team_signals), and the status-dot colours under nearby cut-offs.
Live part (--live): reads the groups in the database backend/.env points at, read-only, and counts what
each nudge changes. Run from backend/:  python -m component4.evaluate_thresholds [--live]
"""
import argparse
import json

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from .api import service
from .api.outlook import status
from .common import config
from .m2_prediction.data import split
from .synthetic import generator
from .synthetic.feature_spec import FEATURES, feature_names

SETTINGS = config.THRESHOLD_TESTS
POINTS = config.EXPLAINABILITY["probability_to_points"]


def by_signal_count(model, bundle, x_test, y_test) -> list[dict]:
    rng = np.random.default_rng(config.SEED)
    names = feature_names()
    medians, sample = bundle["medians"], bundle.get("fill_sample", bundle["background"])
    rows = []
    for count in range(1, len(names) + 1):
        aucs, briers, widths = [], [], []
        for subset in range(SETTINGS["subsets_per_count"]):
            missing = [n for n in names if n not in set(rng.choice(names, count, replace=False))]
            filled = x_test.copy()
            for name in missing:
                filled[name] = medians[name]
            probabilities = model.predict_proba(filled[names])[:, 1]
            aucs.append(roc_auc_score(y_test, probabilities))
            briers.append(brier_score_loss(y_test, probabilities))
            if subset < SETTINGS["range_subsets"]:
                for _, row in filled.head(SETTINGS["range_rows"]).iterrows():
                    spread = service.probability_range(model, row, missing, sample)
                    widths.append(spread["high"] - spread["low"])
        rows.append({"measured_signals": count, "auc_roc": round(float(np.mean(aucs)), 4),
                     "brier": round(float(np.mean(briers)), 4), "range_width_points": round(float(np.mean(widths)), 1)})
    return rows


def status_cutoffs(model, x_test, y_test) -> list[dict]:
    probabilities = model.predict_proba(x_test[feature_names()])[:, 1] * POINTS
    out = []
    for rules in SETTINGS["status"]:
        merged = {**config.STATUS_RULES, **rules}
        colours = np.array([status(p, row, merged) for p, row in zip(probabilities, x_test.to_dict("records"))])
        out.append({**rules, "colours": {c: {"share": round(float((colours == c).mean()), 3),
                                             "success_rate": round(float(y_test[colours == c].mean()), 3) if (colours == c).any() else None}
                                         for c in ("good", "fine", "attention")}})
    return out


def live() -> dict:
    import component2.db as c2db
    from .m1_feature_integration.team_sources import team_project

    conn = c2db.connect()
    teams = [t for (t,) in conn.execute("SELECT team_id FROM teams").fetchall()]
    projects = [team_project(conn, t) for t in teams]
    signals = [len(FEATURES) - len(p["missing"]) for p in projects]
    return {
        "groups": len(teams),
        "shown_at_min_signals": {v: sum(s >= v for s in signals) for v in SETTINGS["min_team_signals"]},
        "lagging_trusted_at_items": {v: sum(p["stage_items"] >= v for p in projects) for v in SETTINGS["min_stage_items"]},
        "weak_readiness_areas_below": {v: sum(a["score"] < v for p in projects for a in (p["readiness"] or {}).get("areas", []))
                                       for v in SETTINGS["weak_below"]},
        "recurring_causes_at": {v: sum(n >= v for p in projects for n in p["cause_counts"].values())
                                for v in SETTINGS["recurring_at_least"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="also count changes on the groups in the database")
    args = parser.parse_args()

    bundle = service.load_bundle()
    model = bundle["model"]
    x_test, y_test = split(generator.generate())["test"]
    report = {"by_signal_count": by_signal_count(model, bundle, x_test, y_test),
              "status_cutoffs": status_cutoffs(model, x_test, y_test)}
    if args.live:
        report["live"] = live()
    config.THRESHOLD_TESTS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'signals':>8}{'AUC':>8}{'Brier':>8}{'range (pts)':>13}")
    for row in report["by_signal_count"]:
        print(f"{row['measured_signals']:>8}{row['auc_roc']:>8.3f}{row['brier']:>8.3f}{row['range_width_points']:>13.1f}")
    for row in report["status_cutoffs"]:
        print(f"green >= {row['good_at']}, yellow >= {row['fine_at']}: {row['colours']}")
    if args.live:
        print(json.dumps(report["live"], indent=2))
    print(f"Saved to {config.THRESHOLD_TESTS_JSON}")


if __name__ == "__main__":
    main()
