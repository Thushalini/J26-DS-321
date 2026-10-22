"""The shared milestone schedule every SLIIT DS final-year group follows (config.COHORT_SCHEDULE), counted in
months from each group's own project start, merged with the group's own milestones from Component 1.

A group milestone with the same name as a schedule milestone replaces it: the group's date wins, and it keeps
the schedule's expected stages unless Component 1 gave its own. For now the schedule is entered by hand; a later
update would let a coordinator maintain it per intake.
"""
import calendar
from datetime import date

from ..common import config


def add_months(day: date, months: int) -> date:
    index = day.month - 1 + months
    year, month = day.year + index // 12, index % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def cohort(start: str) -> list[dict]:
    first = date.fromisoformat(start)
    return [{"name": m["name"], "date": add_months(first, m["months"]).isoformat(), "stages": list(m["stages"]),
             "work": m.get("work"), "source": "schedule"} for m in config.COHORT_SCHEDULE["milestones"]]


def _key(name: str | None) -> str:
    return (name or "").strip().lower()


def merge(group: list[dict], shared: list[dict]) -> list[dict]:
    remaining = {_key(m["name"]): m for m in shared}
    combined = []
    for m in group:
        match = remaining.pop(_key(m["name"]), None)
        stages = match["stages"] if match and m.get("paced") else m["stages"]
        combined.append({"name": m["name"], "date": m["date"], "stages": stages,
                         "work": match.get("work") if match else None, "source": "group"})
    return sorted(combined + list(remaining.values()), key=lambda m: m["date"])
