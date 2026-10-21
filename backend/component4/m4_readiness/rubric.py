"""Module 4: MLOps readiness, scored from the file names in the team's repository.

Each area has criteria; a criterion is met when a file matching one of its patterns exists. Only file
names are read, never their contents, so "not found" means not visible in the repo, not "not done".
The criteria in config.READINESS are a draft until approved.
"""
import re

from ..common import config


def normalise(files: list[str]) -> dict[str, str]:
    return {path.replace("\\", "/").lower(): path for path in sorted(set(files))}


def first_match(patterns: list[str], paths: dict[str, str]) -> str | None:
    for lowered, original in paths.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            return original
    return None


def check(criteria: dict, paths: dict[str, str]) -> dict:
    found, missing, earned, possible = [], [], 0, 0
    for criterion in criteria.values():
        possible += criterion["points"]
        example = first_match(criterion["patterns"], paths)
        if example:
            earned += criterion["points"]
            found.append({"text": criterion["text"], "file": example})
        else:
            missing.append(criterion["text"])
    return {"share": earned / possible, "found": found, "missing": missing}


def _reason(found: list[dict], missing: list[str]) -> str:
    text = config.READINESS["text"]
    parts = []
    if found:
        parts.append(text["found"].format(items=", ".join(item["text"] for item in found)))
    if missing:
        parts.append(text["missing"].format(items=", ".join(missing)))
    return " ".join(parts)


def assess(files: list[str]) -> dict | None:
    if not files:
        return None
    settings = config.READINESS
    paths = normalise(files)
    areas = []
    for area_id, area in settings["areas"].items():
        result = check(area["criteria"], paths)
        areas.append({"id": area_id, "label": area["label"], "stage": area["stage"],
                      "score": round(settings["max_score"] * result["share"]),
                      "reason": _reason(result["found"], result["missing"]),
                      "found": result["found"], "missing": result["missing"]})
    return {"score": round(sum(a["score"] for a in areas) / len(areas)), "areas": areas,
            "files_checked": len(paths), "status": settings["status"]}
