"""What the Component 3 routes do, kept apart from FastAPI so it can be tested on its own."""
import json

from . import store

LEVEL_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _missing_table(error):
    return "component3_alerts" in str(error)      # 'no such table' (SQLite) / 'does not exist' (PostgreSQL)


def _public(row):
    return {
        "team_id": row["team_id"], "run_id": row["run_id"], "created_at": row["created_at"],
        "run_ended_at": row["run_ended_at"], "anomaly_score": row["anomaly_score"],
        "severity_score": row["severity_score"], "severity_level": row["severity_level"],
        "is_issue": bool(row["is_issue"]), "resolved": bool(row["resolved"]),
        "resolved_at": row["resolved_at"], "title": row["title"], "evidence": row["evidence"],
        "recommended_cause": row["recommended_cause"], "second_cause": row["second_cause"],
        "recommendation": row["recommendation"], "n_epochs": row["n_epochs"],
        "drift_severity_score": row["drift_severity_score"], "drift_source": row["drift_source"],
        "root_cause_probabilities": json.loads(row["cause_probabilities"] or "{}"),
    }


def alerts_payload(conn, team_id):
    """All scored runs of a team: open issues first (worst first), then resolved, then healthy runs."""
    try:
        rows = store.list_alerts(conn, team_id)
    except Exception as e:
        if not _missing_table(e):
            raise
        rows = []                                  # table not created yet = no alerts yet
    alerts = [_public(r) for r in rows]

    def order(a):
        group = 0 if a["is_issue"] and not a["resolved"] else 1 if a["is_issue"] else 2
        return (group, LEVEL_ORDER.get(a["severity_level"], 3), -(a["severity_score"] or 0))

    alerts.sort(key=order)
    issues = [a for a in alerts if a["is_issue"]]
    return {"team_id": team_id, "alerts": alerts,
            "counts": {"open": sum(1 for a in issues if not a["resolved"]),
                       "resolved": sum(1 for a in issues if a["resolved"]),
                       "healthy": sum(1 for a in alerts if not a["is_issue"])}}


def resolve_alert(conn, team_id, run_id, resolved, user_id=None):
    """The updated alert, or None if this team has no such run."""
    try:
        if not store.set_resolved(conn, team_id, run_id, resolved, user_id):
            return None
    except Exception as e:
        if _missing_table(e):
            return None
        raise
    return next((a for a in alerts_payload(conn, team_id)["alerts"] if a["run_id"] == run_id), None)


NEUTRAL_GPU, NEUTRAL_COMMITS = 66.0, 2.0
CONTEXT_DEFAULTS = {"hyperparam_tuning_logged": 0, "commit_gap_days": 1.0, "notebook_edit_freq_per_day": 3.0}


def complete_run(team_id, run):
    """A reporter only knows the training numbers. Fill in what it cannot know, and list what was assumed:
    GPU use and commits get a neutral value; the drift score comes from Component 2's table for this team
    (0.0 when it has none); the other context facts get neutral defaults."""
    run = dict(run)
    assumed = list(run.get("assumed", []))
    epochs = []
    for e in run.get("epochs", []):
        e = dict(e)
        if e.get("gpu_utilization_pct") is None:
            e["gpu_utilization_pct"] = NEUTRAL_GPU
            if "gpu_utilization_pct (not measured)" not in assumed:
                assumed.append("gpu_utilization_pct (not measured)")
        if e.get("commits_that_day") is None:
            e["commits_that_day"] = NEUTRAL_COMMITS
            if "commits_that_day (neutral value)" not in assumed:
                assumed.append("commits_that_day (neutral value)")
        epochs.append(e)
    run["epochs"] = epochs
    ctx = dict(run.get("context") or {})
    if "drift_severity_score" not in ctx:
        from ..collector import drift_reader                  # Component 2's drift scores for this team
        d = drift_reader.read_drift(team_id)
        if d["drift_severity_score"] is None:
            ctx["drift_severity_score"], ctx["drift_source"] = 0.0, "default_0_no_score"
        else:
            ctx["drift_severity_score"], ctx["drift_source"] = d["drift_severity_score"], "component2"
    for k, v in CONTEXT_DEFAULTS.items():
        if k not in ctx:
            ctx[k] = v
            assumed.append(f"{k} (default)")
    run["context"], run["assumed"] = ctx, assumed
    return run


def ingest_run(conn, team_id, run, run_ended_at=None):
    """Score one run (same input as predict_live.predict_run, but the context and the GPU / commit numbers
    may be missing: complete_run fills them) and save the alert for the team.
    Raises ValueError for a badly formed run."""
    from ..predict_live import predict_run          # loads the trained models, so only when needed
    result = predict_run(complete_run(team_id, run))
    store.ensure_schema(conn)
    store.save_alert(conn, team_id, result, run_ended_at)
    return result
