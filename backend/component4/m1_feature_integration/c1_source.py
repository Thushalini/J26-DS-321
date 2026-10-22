"""Component 1 (Senadheera): the team's blueprint, read from its saved events. Nothing is written.

Gives novelty and feasibility (empty until Component 1 computes them), the objective and metric (for the
performance target), the project deadline and the named milestones. Component 1 doesn't say which lifecycle
stages a milestone expects, so until it does, a milestone expects the stages an even pace would have reached
by its date (config.TIMELINE["pace_order"]), counted from the project start (project_start).
"""
import math
from datetime import date

from ..common import config


def snapshot(conn, team_id: str) -> dict | None:
    from component1.common.events import EventStore
    from component1.plan import plan_from_events
    from component1.storage import load_events

    events = load_events(conn, team_id)
    if not events:
        return None
    plan = plan_from_events(team_id, events)
    timeline = next((f.get("timeline") for f in plan["fields"] if f["name"] == "timeline"), None) or {}
    values = {f["name"]: f.get("value") for f in plan["fields"]}
    return {"blueprint": EventStore.from_events(team_id, events).state.model_dump(mode="json"),
            "deadline": timeline.get("deadline"), "milestones": plan.get("milestones") or [],
            "objective": values.get("objective"), "metric": values.get("evaluation_metric")}


def project_start(*candidates: str | None) -> str | None:
    """The first known of: the repo's creation date, the first commit, the group's creation (YYYY-MM-DD)."""
    return next((c[:10] for c in candidates if c), None)


def expected_stages(when: str, start: str, deadline: str) -> list[str]:
    order = config.TIMELINE["pace_order"]
    span = (date.fromisoformat(deadline) - date.fromisoformat(start)).days
    if span <= 0:
        return list(order)
    share = (date.fromisoformat(when) - date.fromisoformat(start)).days / span
    return order[:max(1, min(len(order), math.ceil(share * len(order))))]


def milestones(snap: dict, start: str | None, fallback_deadline: str | None = None) -> list[dict]:
    """Dated milestones plus the deadline, each with the stages that should show work by then. Without a
    deadline from the blueprint, the even pace runs to fallback_deadline (the shared schedule's end)."""
    deadline = snap.get("deadline")
    pace_end = deadline or fallback_deadline
    out = []
    for m in snap.get("milestones", []):
        if not m.get("date"):
            continue
        paced = not m.get("stages")
        stages = (expected_stages(m["date"], start, pace_end) if start and pace_end else []) if paced else m["stages"]
        out.append({"name": m.get("name"), "date": m["date"], "stages": stages, "paced": paced})
    if deadline:
        out.append({"name": config.TIMELINE["deadline_label"], "date": deadline, "stages": list(config.TIMELINE["pace_order"])})
    return out
