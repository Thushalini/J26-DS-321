"""Collects a real team's signals from the other components' saved outputs in the shared database.

Component 2: behaviour summary and member roles. Component 3: the team's scored runs (alerts).
Module 4: MLOps readiness from the group's repo: key files read from GitHub (cached), else synced file names.
Project health: reproducibility (repo files), data quality (C3 leakage + validation files) and model
performance (C3's best validation score against the blueprint's target).
Component 2's lagging stages are only trusted once the window has enough labelled work to judge stages.
Component 1: novelty and feasibility from the blueprint (missing until Component 1 computes them). Timeline
adherence uses the shared SLIIT DS schedule (from the project start) plus the blueprint's own deadline and
milestones; research-dataset teams skip the shared schedule. Anything missing is filled with typical values.
"""
import json
from collections import Counter
from datetime import datetime, timezone

from ..common import config
from ..m4_readiness import repo_scan
from ..m4_readiness.rubric import assess
from . import c1_source, project_health, schedule, timeline
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


def team_project(conn, team_id: str, allow_network: bool = True) -> dict:
    """allow_network=False only uses a cached repo scan (for lists and read-only scripts)."""
    from component2.api import service as c2
    from component3.api import service as c3

    settings = config.TEAM_SOURCES
    summary = _safe(lambda: c2.team_summary(conn, team_id, settings["c2_window_days"]))
    roles = _safe(lambda: c2.member_roles(conn, team_id, settings["c2_roles_window"]))
    alerts = (_safe(lambda: c3.alerts_payload(conn, team_id)) or {}).get("alerts") or None
    repo_url, last_synced, created_at, is_research = _safe(lambda: conn.execute(
        "SELECT repo_url, last_synced, created_at, is_research FROM teams WHERE team_id = ?",
        (team_id,)).fetchone()) or (None, None, None, None)
    scan = _safe(lambda: repo_scan.get(conn, team_id, repo_url, last_synced, allow_network=allow_network))
    files = scan["files"] if scan else _safe(lambda: repo_files(conn, team_id)) or []
    evidence = scan["evidence"] if scan else None
    readiness = assess(files, evidence)
    metric = (_safe(lambda: conn.execute("SELECT metric FROM blueprints WHERE team_id = ?", (team_id,)).fetchone()) or [None])[0]

    plan = _safe(lambda: c1_source.snapshot(conn, team_id)) or {}
    metric_text = " ".join(x for x in (metric, plan.get("metric")) if x)
    health = project_health.features(alerts, files, metric_text, evidence, plan.get("objective"))
    first_commit = (_safe(lambda: conn.execute("SELECT MIN(ts) FROM activities WHERE team_id = ? AND source = ?",
                                               (team_id, "commit")).fetchone()) or [None])[0]
    starts = {"repo": (evidence or {}).get("repo_created_at"), "first_commit": first_commit, "group": created_at}
    start = c1_source.project_start(*starts.values())
    shared = schedule.cohort(start) if start and not is_research else []
    milestones = schedule.merge(
        c1_source.milestones(plan, start, max(m["date"] for m in shared) if shared else None) if plan else [], shared)
    if milestones:
        daily = _safe(lambda: c2.daily_stage_counts(conn, team_id, config.TIMELINE["history_days"])) or []
        today = datetime.now(timezone.utc).date().isoformat()
        health["timeline_adherence"] = timeline.adherence(milestones, daily, alerts, today)
        for m in milestones:
            m["due"] = m["date"] <= today
            m["met"] = timeline.met(m, daily, alerts) if m["due"] and m["stages"] else None
    project = assemble(blueprint=plan.get("blueprint"), c2_summary=summary, c2_roles=roles, c3_runs=alerts,
                       readiness=readiness, project_health=health)
    project["milestones"] = milestones
    project["timeline_basis"] = {"start": start, "start_from": next((k for k, v in starts.items() if v), None),
                                 "schedule": config.COHORT_SCHEDULE["name"] if shared else None,
                                 "schedule_status": config.COHORT_SCHEDULE["status"] if shared else None}
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
