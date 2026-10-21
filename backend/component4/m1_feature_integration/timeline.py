"""Timeline adherence: share of the milestones already due whose expected stages show real work by their date.

Evidence comes only from Component 2 (labelled work per stage per day) and, for model stages, Component 3's
runs; nothing is ticked by hand. Expects milestones as [{"name", "date": "YYYY-MM-DD", "stages": [...]}], the
format asked of Component 1. Missing until the first milestone with stages is due.
"""
from ..common import config


def met(milestone: dict, daily_counts: list[dict], runs: list[dict] | None) -> bool:
    settings = config.TIMELINE
    for stage in milestone["stages"]:
        items = sum(day.get(stage, 0) for day in daily_counts if day["date"] <= milestone["date"])
        if stage in settings["run_stages"]:
            items += sum(1 for run in runs or [] if (run.get("created_at") or "")[:10] <= milestone["date"])
        if items < settings["min_items_per_stage"]:
            return False
    return True


def adherence(milestones: list[dict] | None, daily_counts: list[dict], runs: list[dict] | None, today: str) -> float | None:
    due = [m for m in milestones or [] if m.get("stages") and m["date"] <= today]
    if not due:
        return None
    return round(sum(met(m, daily_counts, runs) for m in due) / len(due), 3)
