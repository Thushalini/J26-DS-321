"""Make the result figures for the report and slides (test runs only).

Run from the REPOSITORY ROOT, after training (python -m backend.component3.train_pipeline):

    python -m backend.component3.scripts.make_figures

Needs matplotlib (pip install matplotlib). Saves PNG files to backend/component3/results/figures/:
  fig1..fig5            simulated test runs
  real_fig1..real_fig5  real test runs (the same figures)
"""
import matplotlib

matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from ..common import config

BLUE, RED, AMBER, GREEN, GREY = "#2a78d6", "#d03b3b", "#fab219", "#0ca30c", "#898781"
OUT = config.RESULTS_DIR / "figures"
CAUSE_NAMES = ["Overfitting", "Stalled", "Data leakage", "Resource\nbottleneck", "Premature\nevaluation"]
SOURCES = {"simulated": ("", "simulated test runs", config.RUN_LABELS_CSV, "ablation_results.csv"),
           "real": ("real_", "real test runs", config.REAL_RUN_LABELS_CSV, "ablation_results_real.csv")}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 100})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / name)


def load(source):
    _, _, label_csv, ablation_csv = SOURCES[source]
    labels = pd.read_csv(label_csv)[["run_id"] + config.LABELS]
    probs = pd.read_csv(config.RESULTS_DIR / "root_cause_probs.csv").merge(labels, on="run_id")
    sev = pd.read_csv(config.RESULTS_DIR / "intervention_results.csv")
    abl = pd.read_csv(config.RESULTS_DIR / ablation_csv)
    pick = lambda d: d[(d["split"] == "test") & (d["source"] == source)]
    return pick(probs), pick(sev), abl


def roc_figure(test, prefix, label):
    y, s = test["is_anomaly"], test["anomaly_score"]
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"Isolation Forest (AUC = {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Random guess (AUC = 0.5)")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02),
           title=f"Stage 1: anomaly detection\n({label})")
    ax.legend(loc="lower right", frameon=False)
    save(fig, f"{prefix}fig1_roc_anomaly_detection.png")


def confusion_figure(test, prefix, label):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    for i, ax in enumerate(axes):
        y = test[config.LABELS[i]].to_numpy()
        p = (test[config.PROB_COLS[i]].to_numpy() >= 0.5).astype(int)
        cm = confusion_matrix(y, p, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["No", "Yes"])
        ax.set_yticks([0, 1], ["No", "Yes"])
        ax.set_xlabel("Predicted")
        if i == 0:
            ax.set_ylabel("Actual")
        ax.set_title(CAUSE_NAMES[i] + ("\n(no real examples)" if y.sum() == 0 else ""), fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"Stage 2: confusion matrix for each root cause ({label}, threshold 0.5)", fontweight="bold", y=1.06)
    save(fig, f"{prefix}fig2_confusion_matrices.png")


def calibration_figure(test, prefix, label):
    p = test[config.PROB_COLS].to_numpy().ravel()
    y = test[config.LABELS].to_numpy().ravel()
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfectly calibrated")
    ax.plot(xs, ys, "o-", color=BLUE, lw=2, label="Platt-scaled model")
    for x, yy, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, yy), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=GREY)
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1.05),
           title=f"Stage 2: probability calibration\n({label})")
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.04, "numbers = predictions in each bin", ha="right", fontsize=8, color=GREY, transform=ax.transAxes)
    save(fig, f"{prefix}fig3_calibration.png")


def comparison_figure(abl, prefix, label):
    short = {"Baseline A (generic rules)": "Rules A\n(generic)",
             "Baseline B (best single rule per cause, fit on train)": "Rules B\n(tuned)",
             "Trained ML (full)": "Trained ML\n(full)", "ML without anomaly score": "ML w/o\nanomaly",
             "ML without drift score": "ML w/o\ndrift"}
    names = [short.get(s, s) for s in abl["system"]]
    cols = [GREY, GREY, BLUE, "#9bbfec", "#9bbfec"]
    panels = [("macro_F1", "Macro F1\n(higher is better)", "{:.2f}"), ("hamming_loss", "Hamming loss\n(lower is better)", "{:.3f}"),
              ("healthy_flagged", "Healthy runs wrongly flagged\n(lower is better)", "{:.0%}")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (col, title, fmt) in zip(axes, panels):
        vals = abl[col].astype(float)
        bars = ax.bar(names, vals, color=cols)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_ylim(0, max(vals.max(), 1e-9) * 1.18)
    fig.suptitle(f"Trained models vs rule-based baselines ({label})", fontweight="bold", y=1.10)
    save(fig, f"{prefix}fig4_baseline_vs_ml.png")


def severity_figure(sev, prefix, label):
    order = ["healthy", "overfitting", "stalled", "data_leakage", "resource_bottleneck", "premature_evaluation"]
    ct = pd.crosstab(sev["run_type"], sev["level"]).reindex(order).dropna(how="all").fillna(0)
    for lv in ["HIGH", "MEDIUM", "LOW"]:
        if lv not in ct:
            ct[lv] = 0
    fig, ax = plt.subplots(figsize=(8, 4))
    left = np.zeros(len(ct))
    for lv, colr in [("HIGH", RED), ("MEDIUM", AMBER), ("LOW", GREEN)]:
        ax.barh(ct.index.str.replace("_", " "), ct[lv], left=left, color=colr, label=lv)
        for i, (v, l0) in enumerate(zip(ct[lv], left)):
            if v >= 5:
                ax.text(l0 + v / 2, i, int(v), ha="center", va="center", color="white", fontweight="bold")
            elif v > 0:
                ax.text(l0 + v + 0.6, i, int(v), ha="left", va="center", color=colr, fontweight="bold")
        left += ct[lv].to_numpy()
    ax.invert_yaxis()
    ax.set(xlabel="Number of test runs", title=f"Stage 3: severity level for each type of run ({label})")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    save(fig, f"{prefix}fig5_severity_by_run_type.png")


def main():
    have_real = config.REAL_RUN_LABELS_CSV.exists()
    for source, (prefix, label, _, _) in SOURCES.items():
        if source == "real" and not have_real:
            continue
        test, sev, abl = load(source)
        if test.empty:
            continue
        roc_figure(test, prefix, label)
        confusion_figure(test, prefix, label)
        calibration_figure(test, prefix, label)
        comparison_figure(abl, prefix, label)
        severity_figure(sev, prefix, label)


if __name__ == "__main__":
    main()
