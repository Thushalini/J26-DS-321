"""Recurring failure patterns: clusters Component 3's issue history by root-cause probabilities.

Each issue is a run Component 3 rated MEDIUM or HIGH. Issues with a similar mix of root causes land in
the same pattern (k-means, number of patterns chosen by silhouette score). A team "has" a recurring
pattern when several of its issues fall into it, which is the signal the improvement engine acts on.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..common import config
from ..m1_feature_integration.c3_adapter import _is_issue

SETTINGS = config.FAILURE_PATTERNS


def issues_frame(runs: list[dict]) -> pd.DataFrame:
    """C3 live results -> one row per issue with its cause probabilities, severity and team."""
    rows = []
    for run in runs:
        if not _is_issue(run):
            continue
        probabilities = run["root_cause_probabilities"]
        rows.append({"run_id": run.get("run_id"), "team_id": run.get("team_id"),
                     "severity_score": float(run["severity_score"]),
                     **{cause: float(probabilities.get(cause, 0.0)) for cause in SETTINGS["causes"]}})
    return pd.DataFrame(rows, columns=["run_id", "team_id", "severity_score", *SETTINGS["causes"]])


def _label(centre: np.ndarray) -> list[str]:
    present = [cause for cause, p in zip(SETTINGS["causes"], centre) if p >= SETTINGS["cause_present_at"]]
    return present or [SETTINGS["causes"][int(np.argmax(centre))]]


def find_patterns(issues: pd.DataFrame, seed: int = config.SEED) -> dict:
    values = issues[SETTINGS["causes"]].to_numpy()
    candidates = [k for k in SETTINGS["k_range"] if k < len(issues)]
    if not candidates:
        return {"k": 0, "silhouette": None, "patterns": [], "assignments": []}

    scored = {}
    for k in candidates:
        labels = KMeans(n_clusters=k, n_init=SETTINGS["n_init"], random_state=seed).fit_predict(values)
        if len(set(labels)) > 1:
            scored[k] = silhouette_score(values, labels)
    k = max(scored, key=scored.get) if scored else candidates[0]
    model = KMeans(n_clusters=k, n_init=SETTINGS["n_init"], random_state=seed).fit(values)

    patterns = []
    for cluster, centre in enumerate(model.cluster_centers_):
        members = issues[model.labels_ == cluster]
        patterns.append({
            "pattern_id": cluster,
            "causes": _label(centre),
            "issues": int(len(members)),
            "share": round(len(members) / len(issues), 3),
            "mean_severity": round(float(members["severity_score"].mean()), 3),
            "mean_probabilities": {c: round(float(p), 3) for c, p in zip(SETTINGS["causes"], centre)},
        })
    patterns.sort(key=lambda p: p["issues"], reverse=True)
    return {"k": int(k), "silhouette": round(float(scored.get(k, float("nan"))), 3) if scored else None,
            "patterns": patterns, "assignments": model.labels_.tolist()}


def recurring_for_team(issues: pd.DataFrame, assignments: list[int], patterns: list[dict], team_id) -> list[dict]:
    mine = issues.assign(pattern=assignments)
    mine = mine[mine["team_id"] == team_id]
    counts = mine["pattern"].value_counts()
    by_id = {p["pattern_id"]: p for p in patterns}
    return [
        {"pattern_id": int(pid), "causes": by_id[pid]["causes"], "issues": int(n)}
        for pid, n in counts.items() if n >= SETTINGS["recurring_at_least"]
    ]


def issues_from_c3_results(results_dir=config.C3_RESULTS_DIR) -> pd.DataFrame:
    """Component 3's saved batch results (its synthetic runs) in the same shape as issues_frame()."""
    levels = pd.read_csv(results_dir / "intervention_results.csv")[["run_id", "severity", "level"]]
    probs = pd.read_csv(results_dir / "root_cause_probs.csv")
    merged = levels.merge(probs, on="run_id")
    runs = [{"run_id": r["run_id"], "severity_level": r["level"], "severity_score": r["severity"],
             "root_cause_probabilities": {c: r[f"prob_{c}"] for c in SETTINGS["causes"]}}
            for r in merged.to_dict("records")]
    return issues_frame(runs)
