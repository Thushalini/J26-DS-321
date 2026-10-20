"""Component 1 adapter: the Living DS Project Blueprint -> novelty and feasibility features.

From Senadheera's schema (backend/component1/common/schema.py): novelty_score is 0-1 or None, and
feasibility is a set of checks, each "pass", "warning" or "fail" (or None before any check ran).
Feasibility becomes one 0-1 score: the average of the checks' points from config.
"""
from ..common import config


def feasibility_score(checks: dict | None) -> float | None:
    if not checks:
        return None
    points = config.C1["feasibility_points"]
    known = [points[result] for result in checks.values() if result in points]
    return round(sum(known) / len(known), 4) if known else None


def project_features(blueprint: dict) -> dict:
    fields = blueprint.get("fields") or {}
    settled = config.C1["settled_field_statuses"]
    statuses = [state.get("status") for state in fields.values() if isinstance(state, dict)]
    novelty = blueprint.get("novelty_score")
    return {
        "features": {
            "novelty_score": None if novelty is None else float(novelty),
            "feasibility_score": feasibility_score(blueprint.get("feasibility")),
        },
        "context": {
            "blueprint_version": blueprint.get("version"),
            "feasibility_checks": blueprint.get("feasibility") or {},
            "fields_settled": sum(status in settled for status in statuses),
            "fields_total": len(statuses),
        },
    }
