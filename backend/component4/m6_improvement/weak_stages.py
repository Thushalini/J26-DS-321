"""Which lifecycle stages need help, found from the data instead of chosen by hand.

Four sources: Component 2's lagging stages (little recent work), Component 3's recurring issue causes
(mapped to the stages that fix them), the stages behind this prediction's top recommendations, and
weak MLOps readiness areas.
"""
from ..common import config


def weak_stages(lagging: list[str], recurring_causes: list[str], recommendations: list[dict],
                readiness: dict | None = None) -> list[dict]:
    found: dict[str, list[str]] = {}

    def add(stage, source):
        if stage in config.STAGE_LABELS and source not in found.setdefault(stage, []):
            found[stage].append(source)

    for stage in lagging:
        add(stage, "behavior")
    for cause in recurring_causes:
        for stage in config.CAUSE_STAGES.get(cause, []):
            add(stage, "issues")
    for recommendation in recommendations:
        add(recommendation.get("stage"), "prediction")
    for area in (readiness or {}).get("areas", []):
        if area["score"] < config.READINESS["weak_below"]:
            add(area["stage"], "readiness")
    return [{"stage": stage, "label": config.STAGE_LABELS[stage], "sources": sources} for stage, sources in found.items()]
