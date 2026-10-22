
"""Component 3 API service: training-run scoring, Git features, and alerts."""

import json
from datetime import datetime, timedelta, timezone

from . import store

LEVEL_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

NEUTRAL_GPU = 66.0
NEUTRAL_COMMITS = 2.0

GIT_WINDOW_DAYS = 7

CONTEXT_DEFAULTS = {
    "hyperparam_tuning_logged": 0,
    "commit_gap_days": 1.0,
    "notebook_edit_freq_per_day": 3.0,
}


def _missing_table(error):
    return "component3_alerts" in str(error)


def _public(row):
    return {
        "team_id": row["team_id"],
        "run_id": row["run_id"],
        "created_at": row["created_at"],
        "run_ended_at": row["run_ended_at"],
        "anomaly_score": row["anomaly_score"],
        "severity_score": row["severity_score"],
        "severity_level": row["severity_level"],
        "is_issue": bool(row["is_issue"]),
        "resolved": bool(row["resolved"]),
        "resolved_at": row["resolved_at"],
        "title": row["title"],
        "evidence": row["evidence"],
        "recommended_cause": row["recommended_cause"],
        "second_cause": row["second_cause"],
        "recommendation": row["recommendation"],
        "n_epochs": row["n_epochs"],
        "drift_severity_score": row["drift_severity_score"],
        "drift_source": row["drift_source"],
        "root_cause_probabilities": json.loads(
            row["cause_probabilities"] or "{}"
        ),
    }


def alerts_payload(conn, team_id):
    """List a team's scored runs, with unresolved issues first."""

    try:
        rows = store.list_alerts(conn, team_id)
    except Exception as e:
        if not _missing_table(e):
            raise
        rows = []

    alerts = [_public(r) for r in rows]

    def order(a):
        group = (
            0 if a["is_issue"] and not a["resolved"]
            else 1 if a["is_issue"]
            else 2
        )
        return (
            group,
            LEVEL_ORDER.get(a["severity_level"], 3),
            -(a["severity_score"] or 0),
        )

    alerts.sort(key=order)

    issues = [a for a in alerts if a["is_issue"]]

    return {
        "team_id": team_id,
        "alerts": alerts,
        "counts": {
            "open": sum(
                1 for a in issues if not a["resolved"]
            ),
            "resolved": sum(
                1 for a in issues if a["resolved"]
            ),
            "healthy": sum(
                1 for a in alerts if not a["is_issue"]
            ),
        },
    }


def resolve_alert(conn, team_id, run_id, resolved, user_id=None):
    """Resolve or reopen an alert belonging to the team."""

    try:
        if not store.set_resolved(
            conn, team_id, run_id, resolved, user_id
        ):
            return None
    except Exception as e:
        if _missing_table(e):
            return None
        raise

    return next(
        (
            a for a in alerts_payload(conn, team_id)["alerts"]
            if a["run_id"] == run_id
        ),
        None,
    )


def _parse_utc(value):
    """Convert ISO timestamps to timezone-aware UTC datetimes."""

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(
            value.strip().replace("Z", "+00:00")
        )
    else:
        raise ValueError("Invalid timestamp")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def _load_git_activity(conn, team_id, epochs):
    """Calculate real Git features from the existing Buddy.ds database.

    Returns None if:
      - Epoch timestamps are missing or invalid.
      - The team has no linked repository.
      - The repository has not completed a successful sync.
      - The latest sync predates the training run.
    """

    if not epochs:
        return None

    if any(not e.get("timestamp") for e in epochs):
        return None

    try:
        epoch_times = [
            _parse_utc(e["timestamp"])
            for e in epochs
        ]
    except (ValueError, TypeError, OverflowError):
        return None

    end_time = max(epoch_times)

    team = conn.execute(
        """
        SELECT repo_url, sync_status, last_synced
        FROM teams
        WHERE team_id = ?
        """,
        (team_id,),
    ).fetchone()

    if team is None:
        return None

    repo_url, sync_status, last_synced = team

    if not repo_url or sync_status != "done" or not last_synced:
        return None

    try:
        sync_time = _parse_utc(last_synced)
    except (ValueError, TypeError, OverflowError):
        return None

    # Only use a repository snapshot taken after training.
    if sync_time < end_time:
        return None

    start_time = end_time - timedelta(
        days=GIT_WINDOW_DAYS
    )

    rows = conn.execute(
        """
        SELECT ts, meta
        FROM activities
        WHERE team_id = ?
          AND source = 'commit'
        """,
        (team_id,),
    ).fetchall()

    daily_commits = {}
    notebook_commits = 0
    latest_commit = None

    for ts, meta_text in rows:
        try:
            commit_time = _parse_utc(ts)
        except (ValueError, TypeError, OverflowError):
            continue

        # Never use commits made after the training run.
        if commit_time > end_time:
            continue

        if latest_commit is None or commit_time > latest_commit:
            latest_commit = commit_time

        if commit_time < start_time:
            continue

        day = commit_time.date().isoformat()

        daily_commits[day] = (
            daily_commits.get(day, 0) + 1
        )

        try:
            meta = json.loads(meta_text or "{}")
        except (TypeError, ValueError):
            meta = {}

        files = meta.get("files", [])

        if not isinstance(files, list):
            files = []

        # Count commits that changed notebook files.
        # Do not count individual notebook cells as edits.
        if any(
            isinstance(path, str)
            and path.lower().endswith(".ipynb")
            for path in files
        ):
            notebook_commits += 1

    if latest_commit is None:
        commit_gap_days = float(GIT_WINDOW_DAYS)
    else:
        gap = (
            end_time - latest_commit
        ).total_seconds() / 86400.0

        commit_gap_days = min(
            float(GIT_WINDOW_DAYS),
            max(0.0, gap),
        )

    return {
        "daily_commits": daily_commits,
        "commit_gap_days": commit_gap_days,
        "notebook_edit_freq_per_day": (
            notebook_commits / float(GIT_WINDOW_DAYS)
        ),
        "last_synced": last_synced,
    }


def complete_run(team_id, run, conn=None):
    """Complete training metrics using actual Git data where available.

    Keep existing values supplied by the reporter.

    When real Git measurements cannot be obtained, use the
    existing neutral defaults and label them as assumed.
    """

    run = dict(run)
    assumed = list(run.get("assumed", []))

    original_epochs = [
        dict(e)
        for e in run.get("epochs", [])
    ]

    git_activity = None

    if conn is not None:
        git_activity = _load_git_activity(
            conn,
            team_id,
            original_epochs,
        )

    epochs = []

    for e in original_epochs:
        if e.get("gpu_utilization_pct") is None:
            e["gpu_utilization_pct"] = NEUTRAL_GPU

            label = "gpu_utilization_pct (not measured)"
            if label not in assumed:
                assumed.append(label)

        if e.get("commits_that_day") is None:
            if git_activity is not None:
                day = _parse_utc(
                    e["timestamp"]
                ).date().isoformat()

                e["commits_that_day"] = git_activity[
                    "daily_commits"
                ].get(day, 0)
            else:
                e["commits_that_day"] = NEUTRAL_COMMITS

                label = "commits_that_day (neutral value)"
                if label not in assumed:
                    assumed.append(label)

        epochs.append(e)

    run["epochs"] = epochs

    ctx = dict(run.get("context") or {})

    if "drift_severity_score" not in ctx:
        from ..collector import drift_reader

        drift = drift_reader.read_drift(team_id)

        if drift["drift_severity_score"] is None:
            ctx["drift_severity_score"] = 0.0
            ctx["drift_source"] = "default_0_no_score"
        else:
            ctx["drift_severity_score"] = drift[
                "drift_severity_score"
            ]
            ctx["drift_source"] = "component2"

    if git_activity is not None:
        ctx.setdefault(
            "commit_gap_days",
            git_activity["commit_gap_days"],
        )

        ctx.setdefault(
            "notebook_edit_freq_per_day",
            git_activity["notebook_edit_freq_per_day"],
        )

    for key, value in CONTEXT_DEFAULTS.items():
        if key not in ctx:
            ctx[key] = value

            label = f"{key} (default)"
            if label not in assumed:
                assumed.append(label)

    run["context"] = ctx
    run["assumed"] = assumed

    return run


def ingest_run(conn, team_id, run, run_ended_at=None):
    """Score a run with existing ML models and save its alert."""

    from ..predict_live import predict_run

    completed = complete_run(
        team_id,
        run,
        conn=conn,
    )

    result = predict_run(completed)

    store.ensure_schema(conn)

    store.save_alert(
        conn,
        team_id,
        result,
        run_ended_at,
    )

    return result
