"""Collects a real team's signals from the other components' saved outputs in the shared database.

Component 2: behaviour summary and member roles. Component 3: the team's scored runs (alerts).
Module 4: MLOps readiness from the file names in the synced repository (names only, never contents).
Project health: reproducibility (repo files), data quality (C3 leakage + validation files) and model
performance (C3's best validation score against the blueprint's target).
Component 2's lagging stages are only trusted once the window has enough labelled work to judge stages.
Component 1 and timeline adherence don't publish team data yet, so those features stay missing and the
prediction reports them as filled with typical values.
"""
import json
from collections import Counter

from ..common import config
from ..m4_readiness.rubric import assess
from . import project_health
from .assemble import assemble


def _safe(fetch):
    try:
        return fetch()
    except Exception:
        return None


def repo_files(conn, team_id: str) -> list[str]:
    rows = conn.execute("SELECT meta FROM activities WHERE team_id = ? AND source = ?", (team_id, "commit")).fetchall()
    files = set()
    for (meta,) in rows:
        files.update(json.loads(meta or "{}").get("files", []))
    return sorted(files)


def team_project(conn, team_id: str) -> dict:
    from component2.api import service as c2
    from component3.api import service as c3

    settings = config.TEAM_SOURCES
    summary = _safe(lambda: c2.team_summary(conn, team_id, settings["c2_window_days"]))
    roles = _safe(lambda: c2.member_roles(conn, team_id, settings["c2_roles_window"]))
    alerts = (_safe(lambda: c3.alerts_payload(conn, team_id)) or {}).get("alerts") or None
    files = _safe(lambda: repo_files(conn, team_id)) or []
    readiness = assess(files)
    metric = (_safe(lambda: conn.execute("SELECT metric FROM blueprints WHERE team_id = ?", (team_id,)).fetchone()) or [None])[0]

    project = assemble(c2_summary=summary, c2_roles=roles, c3_runs=alerts, readiness=readiness,
                       project_health=project_health.features(alerts, files, metric))
    open_causes = Counter(a.get("recommended_cause") for a in alerts or []
                          if a.get("is_issue") and not a.get("resolved") and a.get("recommended_cause"))
    project["cause_counts"] = dict(open_causes)
    project["recurring_counts"] = {cause: n for cause, n in open_causes.items()
                                   if n >= config.FAILURE_PATTERNS["recurring_at_least"]}
    project["recurring_causes"] = list(project["recurring_counts"])
    stage_items = sum(day.get(stage, 0) for day in _safe(lambda: c2.daily_stage_counts(
        conn, team_id, settings["c2_window_days"])) or [] for stage in config.STAGE_LABELS)
    enough_work = stage_items >= settings["min_stage_items"]
    project["stage_items"] = stage_items
    project["lagging_stages"] = (summary or {}).get("lagging_stages", []) if enough_work else []
    project["readiness"] = readiness
    return project
