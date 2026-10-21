"""Does it work on runs unlike the ones it learned from?

Trains the whole pipeline again WITHOUT some real runs, then tests on exactly those runs:
  random_real       a random 15% of the real runs (the same kinds of tasks and models as in training)
  unseen_tasks      every run on the tasks breast_cancer, wine and iris
  unseen_model      every run of the model type sgd_logreg
and compares the trained pipeline with the best single-threshold rule per cause (Baseline B), which gets
the same training runs. Reports what a team would experience: false alarms, missed failures, right cause.

    python -m backend.component3.evaluation.generalization        (about a minute)

Saves results/generalization.csv.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..common import config, features, real_runs
from ..common.splits import load_or_make_splits
from ..stage1_anomaly import anomaly_model
from ..stage2_root_cause import root_cause_model as rcm
from ..stage3_severity import severity
from . import baseline


def prepare():
    traj = pd.read_csv(config.TRAJECTORIES_CSV)
    runs = pd.read_csv(config.RUN_LABELS_CSV)
    sim = features.build_run_features(traj).merge(runs, on="run_id").merge(load_or_make_splits(runs), on="run_id")
    rtraj, rlabels = real_runs.load_real_runs()
    real = features.build_run_features(rtraj).merge(rlabels, on="run_id")
    return sim, real, features.feature_names(features.build_run_features(traj.head(60)))


def holdouts(real):
    random_ids = real.groupby("run_type", group_keys=False).apply(
        lambda g: g.sample(frac=0.15, random_state=config.SEED))["run_id"]
    return {"random_real": real["run_id"].isin(random_ids),
            "unseen_tasks": real["task"].isin(["breast_cancer", "wine", "iris"]),
            "unseen_model": real["model"] == "sgd_logreg"}


def score_levels(test, a_bundle, rc_bundle):
    t = test.copy()
    t["anomaly_score"] = anomaly_model.score(a_bundle, t)
    probs = rcm.predict_probs(rc_bundle, t)
    sev = [severity.severity_score(a, p) for a, p in zip(t["anomaly_score"], probs)]
    t["level"] = [severity.severity_level(s) for s in sev]
    top = [config.CAUSES[int(np.argmax(p))] for p in probs]
    t["named_ok"] = [tp == rt for tp, rt in zip(top, t["run_type"])]
    return t


def main():
    sim, real, names = prepare()
    rows = []
    for name, mask in holdouts(real).items():
        rest = real[~mask]
        tr, va = train_test_split(rest["run_id"], test_size=0.15, random_state=config.SEED, stratify=rest["run_type"])
        real_part = rest.assign(split=np.where(rest["run_id"].isin(tr), "train", "val"))
        pool = pd.concat([sim[sim["split"] != "test"], real_part], ignore_index=True)
        test = real[mask]

        a_bundle = anomaly_model.fit_anomaly_model(pool, names, verbose=False)
        pool = pool.assign(anomaly_score=anomaly_model.score(a_bundle, pool))
        params = {"n_estimators": 100, "max_depth": 2}          # the settings the main pipeline chose (no new tuning here)
        feats = config.ROOT_CAUSE_FEATURES
        rc_bundle = {"models": rcm.fit_models(pool[pool["split"] == "train"], feats, params),
                     "features": feats, "params": params}
        t = score_levels(test, a_bundle, rc_bundle)
        flagged, healthy = t["level"] != "LOW", t["is_anomaly"] == 0
        rows.append({"held_out": name, "system": "Trained pipeline", "runs": len(t), "healthy": int(healthy.sum()),
                     "healthy_flagged": int(flagged[healthy].sum()), "failures": int((~healthy).sum()),
                     "failures_flagged": int(flagged[~healthy].sum()),
                     "cause_right_when_flagged": int((flagged & ~healthy & t["named_ok"]).sum())})

        train_rows = pool[pool["split"] == "train"]
        pred_b = baseline.baseline_b(train_rows, test)
        fb = pred_b.sum(axis=1) > 0
        h = (test["is_anomaly"] == 0).values
        right_b = int(sum(config.CAUSES[int(np.argmax(p))] == rt for p, rt, f, hh in
                          zip(pred_b, test["run_type"], fb, h) if f and not hh and p.any()))
        rows.append({"held_out": name, "system": "Best single rule per cause (B)", "runs": len(test),
                     "healthy": int(h.sum()), "healthy_flagged": int(fb[h].sum()), "failures": int((~h).sum()),
                     "failures_flagged": int(fb[~h].sum()), "cause_right_when_flagged": right_b})
    out = pd.DataFrame(rows)
    out["false_alarm_rate"] = (out["healthy_flagged"] / out["healthy"]).round(3)
    out["failures_caught_rate"] = (out["failures_flagged"] / out["failures"]).round(3)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(config.RESULTS_DIR / "generalization.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
