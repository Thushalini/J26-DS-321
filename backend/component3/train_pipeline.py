"""Train and evaluate the whole Component 3 pipeline in one go.

Run from the REPOSITORY ROOT (the folder that contains `backend/`):

    python -m backend.component3.train_pipeline                  # simulated + real training runs
    python -m backend.component3.train_pipeline --simulated-only # the first version: simulated runs only

The first version learned only from simulated runs and raised an alert for every real run. Version 2
also learns from real training runs (data/real_runs/, made by scripts/make_real_runs.py), and every
result is reported separately for the simulated and the real test runs.

It trains Stage 1 and Stage 2, runs the baseline/ablation study and the severity engine, then saves:
    models/anomaly_model.joblib, models/root_cause_models.joblib
    results/anomaly_scores.csv, root_cause_probs.csv, intervention_results.csv, metrics.json,
            ablation_results.csv (simulated test), ablation_results_real.csv (real test)
"""
import argparse
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import config, features, real_runs
from .common.splits import load_or_make_splits
from .evaluation import ablation, baseline
from .evaluation.metrics import alert_summary, stage2_metrics
from .stage1_anomaly import anomaly_model
from .stage2_root_cause import root_cause_model as rcm
from .stage3_severity import severity

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulated-only", action="store_true", help="train on the simulated runs only (version 1)")
    args = ap.parse_args()
    use_real = real_runs.available() and not args.simulated_only

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    # ---------------- data ----------------
    traj = pd.read_csv(config.TRAJECTORIES_CSV)
    runs = pd.read_csv(config.RUN_LABELS_CSV)
    split = load_or_make_splits(runs)
    sim_feats = features.build_run_features(traj)
    names = features.feature_names(sim_feats)
    base_cols = ["run_id", "run_type", "is_anomaly"]
    d1 = sim_feats.merge(runs[base_cols], on="run_id").merge(split, on="run_id")
    d1["source"] = "simulated"
    labels = runs.merge(split, on="run_id")
    labels["source"] = "simulated"
    all_traj = traj
    print(f"Simulated runs: {len(runs)} | epoch rows: {len(traj)} | split: {split['split'].value_counts().to_dict()}")
    if use_real:
        rtraj, rlabels = real_runs.load_real_runs()
        rsplit = real_runs.load_or_make_real_splits(rlabels)
        rfeats = features.build_run_features(rtraj)
        d1 = pd.concat([d1, rfeats.merge(rlabels[base_cols], on="run_id").merge(rsplit, on="run_id").assign(source="real")],
                       ignore_index=True)
        labels = pd.concat([labels, rlabels.merge(rsplit, on="run_id").assign(source="real")], ignore_index=True)
        all_traj = pd.concat([traj, rtraj], ignore_index=True)
        print(f"Real runs: {len(rlabels)} | epoch rows: {len(rtraj)} | split: {rsplit['split'].value_counts().to_dict()}")
    sources = ["simulated"] + (["real"] if use_real else [])

    # ---------------- Stage 1: anomaly detection ----------------
    print("\n=== STAGE 1: anomaly detection (Isolation Forest, trained on healthy runs only) ===")
    a_bundle = anomaly_model.fit_anomaly_model(d1, names)
    anomaly_model.save(a_bundle)
    d1["anomaly_score"] = anomaly_model.score(a_bundle, d1)
    print(f"Chosen: {a_bundle['params']}")
    aucs = {}
    for src in sources:
        t = d1[(d1["source"] == src) & (d1["split"] == "test")]
        aucs[src] = float(roc_auc_score(t["is_anomaly"], t["anomaly_score"]))
        print(f"TEST AUC-ROC on {src} runs: {aucs[src]:.4f}   (target > 0.85)")
    d1[["run_id", "run_type", "is_anomaly", "split", "source", "anomaly_score"]] \
        .to_csv(config.RESULTS_DIR / "anomaly_scores.csv", index=False)

    # ---------------- Stage 2: root-cause classifier ----------------
    print("\n=== STAGE 2: root-cause classifier (calibrated Gradient Boosting) ===")
    d2 = labels.merge(d1[["run_id", "anomaly_score"]], on="run_id")
    train, val = d2[d2["split"] == "train"], d2[d2["split"] == "val"]
    rc_bundle = rcm.fit_root_cause_models(train, val)
    rcm.save(rc_bundle)
    print(f"Chosen: {rc_bundle['params']}")
    d2[config.PROB_COLS] = rcm.predict_probs(rc_bundle, d2)
    s2 = {}
    for src in sources:
        t = d2[(d2["source"] == src) & (d2["split"] == "test")]
        s2[src] = stage2_metrics(t, t[config.PROB_COLS].values)
        m = s2[src]
        print(f"TEST on {src} runs: macro F1 {m['macro_F1']:.4f} (target > 0.80) | Hamming loss {m['hamming_loss']:.4f} "
              f"| Brier {m['brier']:.4f} (target < 0.15)")
    d2[["run_id", "run_type", "is_anomaly", "split", "source", "anomaly_score"] + config.PROB_COLS] \
        .to_csv(config.RESULTS_DIR / "root_cause_probs.csv", index=False)

    # ---------------- Baselines + ablation ----------------
    rule_a = baseline.baseline_a(all_traj)
    gains = {}
    for src in sources:
        print(f"\n=== BASELINES + ABLATION ({src} test runs) ===")
        # same training runs for every system; only the test runs differ
        data = d2[(d2["split"] != "test") | (d2["source"] == src)]
        table, ci, per_cause = ablation.run_ablation(data, rule_a, rc_bundle)
        print(table.to_string(index=False))
        f1_ml = float(table.loc[table.system == "Trained ML (full)", "macro_F1"].iloc[0])
        gains[src] = {}
        for key, nm in [("vs_baseline_A", "Baseline A (generic rules)"),
                        ("vs_baseline_B", "Baseline B (best single rule per cause, fit on train)")]:
            f1_b = float(table.loc[table.system == nm, "macro_F1"].iloc[0])
            rel = (f1_ml - f1_b) / max(f1_b, 1e-9)
            gains[src][key] = {"baseline_F1": f1_b, "relative_gain": rel,
                               "target_10pct_met": bool(rel >= 0.10), "bootstrap_95_CI_abs": ci[key]}
            print(f"ML {key}: F1 {f1_b:.4f} -> {f1_ml:.4f} ({100 * rel:+.1f}% relative) "
                  f"| 10% target {'MET' if rel >= 0.10 else 'NOT MET'} "
                  f"| 95% CI of gain {ci[key][0]:+.3f} .. {ci[key][1]:+.3f}")
        print("Per-cause F1:\n" + per_cause.to_string(index=False))
        table.to_csv(config.RESULTS_DIR / ("ablation_results.csv" if src == "simulated" else "ablation_results_real.csv"),
                     index=False)

    # ---------------- Stage 3: severity + intervention ----------------
    print("\n=== STAGE 3: severity + intervention ===")
    rows = []
    for _, r in d2.iterrows():
        p = r[config.PROB_COLS].to_numpy(dtype=float)
        s = severity.severity_score(r["anomaly_score"], p)
        lvl = severity.severity_level(s)
        top, second, text = severity.recommend(lvl, p)
        rows.append({"run_id": r["run_id"], "severity": s, "level": lvl, "recommended_cause": top,
                     "second_cause": second, "recommendation": text,
                     "recommended_correct": bool(top is not None and r[f"cause_{top}"] == 1)})
    sev = d2[["run_id", "run_type", "is_anomaly", "split", "source", "anomaly_score"]].merge(pd.DataFrame(rows), on="run_id")
    sev.to_csv(config.RESULTS_DIR / "intervention_results.csv", index=False)
    alerts = {}
    for src in sources:
        st = sev[(sev["source"] == src) & (sev["split"] == "test")]
        high = st[st["level"] == "HIGH"]
        ok = int(((high["is_anomaly"] == 1) & high["recommended_correct"]).sum())
        alerts[src] = alert_summary(st)
        alerts[src].update({"high_cases": len(high), "high_correct": ok,
                            "high_correct_rate": ok / len(high) if len(high) else float("nan")})
        a = alerts[src]
        print(f"[{src} test runs] false alarms on healthy runs: {a['healthy_flagged']}/{a['healthy']} | "
              f"failures alerted: {a['failures_flagged']}/{a['failures']} | right cause when alerted: "
              f"{a['cause_right_when_flagged']}/{a['failures_flagged']} | alert precision {a['alert_precision']:.1%} | "
              f"HIGH correct: {ok}/{len(high)}")

    metrics = {
        "trained_on": sources,
        "stage1_test_AUC_ROC": aucs["simulated"],                  # (kept for the simulated test set)
        "stage2_test": s2["simulated"],
        "ablation_gains": gains["simulated"],
        "stage3": {"high_cases": alerts["simulated"]["high_cases"], "high_correct": alerts["simulated"]["high_correct"],
                   "correct_rate": alerts["simulated"]["high_correct_rate"]},
        "by_test_set": {src: {"stage1_AUC_ROC": aucs[src], "stage2": s2[src], "alerts": alerts[src],
                              "ablation_gains": gains[src]} for src in sources},
        "chosen_params": {"anomaly": a_bundle["params"], "root_cause": rc_bundle["params"]},
    }
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved models/ and results/ (see metrics.json for the summary).")


if __name__ == "__main__":
    main()
