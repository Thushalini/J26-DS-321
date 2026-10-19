"""Component 3 adapter: turns its per-run results (predict_live.py output) into project-level features.

Agreed with Senthazhan on 2026-10-18:
- an issue is a run whose severity_level is MEDIUM or HIGH (or is_issue, once he adds it);
- send both the worst and the average severity, plus the share of runs that are issues;
- open HIGH issues count separately; resolved defaults to false until a "Mark resolved" record exists;
- team_id and run_ended_at are optional pass-through fields; the recent window needs run_ended_at.
"""
from datetime import datetime, timedelta, timezone

from ..common import config


def _is_issue(run: dict) -> bool:
    if "is_issue" in run:
        return bool(run["is_issue"])
    return run.get("severity_level") in config.C3["issue_levels"]


def _ended_at(run: dict) -> datetime | None:
    value = run.get("run_ended_at")
    if not value:
        return None
    moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def project_features(runs: list[dict], team_id: str | None = None, now: datetime | None = None) -> dict:
    if team_id is not None:
        runs = [run for run in runs if run.get("team_id") in (team_id, None)]

    issues = [run for run in runs if _is_issue(run)]
    severities = [float(run["severity_score"]) for run in runs]
    open_high = [run for run in issues
                 if run.get("severity_level") == config.C3["high_level"] and not run.get("resolved", False)]

    features = {
        "issue_count": len(issues),
        "issue_share": round(len(issues) / len(runs), 4) if runs else 0.0,
        "max_severity": round(max(severities, default=0.0), 4),
        "mean_severity": round(_mean(severities), 4),
        "open_high_issues": len(open_high),
        "mean_anomaly_score": round(_mean([float(run["anomaly_score"]) for run in runs]), 4),
    }

    window_days = config.C3["recent_window_days"]
    ended = [_ended_at(run) for run in runs]
    has_times = bool(runs) and all(moment is not None for moment in ended)
    recent = None
    if has_times:
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=window_days)
        recent = sum(1 for run, moment in zip(runs, ended) if moment >= cutoff and _is_issue(run))

    return {
        "features": features,
        "context": {
            "runs": len(runs),
            "recent_window_days": window_days,
            "recent_issue_count": recent,
            "recent_window_available": has_times,
        },
    }
