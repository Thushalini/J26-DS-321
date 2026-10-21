"""Competency per lifecycle stage, and who could help with a lagging stage.

A student's competency for a stage is the credit- and relevance-weighted average of their grade points
in the modules that build that stage, scaled to 0-1. Too little related coursework counts as "not enough
evidence" rather than as weak. Suggestions name the strongest member per stage and never show marks.
This is the only place academic marks are used; they never reach the success-prediction model.
"""
import json

from ...common import config

SETTINGS = config.COMPETENCY


def load_module_stages(path=None) -> dict:
    return json.loads((path or SETTINGS["module_stages_file"]).read_text(encoding="utf-8"))


def competency(results: list[dict], module_stages: dict | None = None) -> dict:
    modules = (module_stages or load_module_stages())["modules"]
    points = SETTINGS["grade_points"]
    earned, possible = {}, {}
    for row in results:
        stages = modules.get(row["code"], {}).get("stages", {})
        if row["grade"] not in points:
            continue
        for stage, weight in stages.items():
            share = row["credits"] * weight
            earned[stage] = earned.get(stage, 0.0) + points[row["grade"]] * share
            possible[stage] = possible.get(stage, 0.0) + SETTINGS["max_points"] * share
    return {
        stage: {"score": round(earned[stage] / possible[stage], 4),
                "evidence": round(possible[stage] / SETTINGS["max_points"], 2)}
        for stage in possible
    }


def suggest_helpers(lagging_stages: list[str], members: dict[str, dict], stage_labels: dict) -> list[dict]:
    """members: member id -> {"name": ..., "competency": competency(...)}. One suggestion per lagging stage."""
    suggestions = []
    for stage in lagging_stages:
        ranked = sorted(
            ((m["competency"][stage]["score"], member_id) for member_id, m in members.items()
             if stage in m["competency"] and m["competency"][stage]["evidence"] >= SETTINGS["min_evidence"]),
            reverse=True,
        )
        label = stage_labels.get(stage, stage)
        if not ranked:
            suggestions.append({"stage": stage, "stage_label": label, "member_id": None, "member_name": None,
                                "reason": f"No uploaded record has enough coursework related to {label.lower()}."})
            continue
        best = ranked[0][1]
        suggestions.append({
            "stage": stage, "stage_label": label, "member_id": best, "member_name": members[best]["name"],
            "reason": f"Strongest record among the uploaded sheets in modules related to {label.lower()}.",
        })
    return suggestions
