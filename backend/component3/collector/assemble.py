"""Join the pieces from the collectors into one run that predict_live.predict_run() can score.

  tracker reader (MLflow / W&B / exported log)  ->  epochs with a timestamp each
  git_collector.collect_git_activity()          ->  commits per day, commit gap, notebook edits
  Component 2                                   ->  drift_severity_score
  the run's logged parameters                   ->  hyperparam_tuning_logged (0 or 1)
"""
from datetime import datetime, timezone

from .git_collector import to_utc


def _epoch_day(timestamp, utc_offset_hours=0.0):
    """Calendar day (ISO string) of an epoch timestamp: ISO text, unix seconds or milliseconds."""
    if isinstance(timestamp, (int, float)):
        ts = timestamp / 1000.0 if timestamp > 1e11 else float(timestamp)
        dt = datetime.fromtimestamp(ts, timezone.utc)
    else:
        dt = to_utc(timestamp)
    return datetime.fromtimestamp(dt.timestamp() + utc_offset_hours * 3600,
                                  timezone.utc).date().isoformat()


def attach_commits(epochs, git_activity):
    """Copy of `epochs` where each epoch has commits_that_day, taken from Git by the day
    of the epoch's 'timestamp'."""
    offset = git_activity.get("utc_offset_hours", 0.0)
    out = []
    for i, ep in enumerate(epochs):
        if "timestamp" not in ep:
            raise ValueError(f"Epoch {i} has no 'timestamp', so it cannot be matched to Git days.")
        e = dict(ep)
        e["commits_that_day"] = git_activity["daily_commits"].get(_epoch_day(ep["timestamp"], offset), 0)
        out.append(e)
    return out


def build_run(run_id, epochs, git_activity, drift_severity_score, hyperparam_tuning_logged,
              drift_source="provided") -> dict:
    """Input dict for predict_live.predict_run().
    drift_severity_score can be None (Component 2 has no score for the team): 0.0 is used and the
    output says so in 'drift_source'."""
    if drift_severity_score is None:
        drift_severity_score, drift_source = 0.0, "default_0_no_score"
    gap = git_activity["commit_gap_days"]
    if gap is None:                                   # no commit at all before the run ended
        gap = git_activity["window_days"]
    return {
        "run_id": run_id,
        "epochs": attach_commits(epochs, git_activity),
        "context": {
            "drift_severity_score": float(drift_severity_score),
            "drift_source": drift_source,
            "hyperparam_tuning_logged": 1 if hyperparam_tuning_logged else 0,
            "commit_gap_days": float(gap),
            "notebook_edit_freq_per_day": float(git_activity["notebook_edit_freq_per_day"]),
        },
    }
