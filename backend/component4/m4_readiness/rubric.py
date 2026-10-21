"""Module 4: MLOps readiness, scored from the file names in the team's repository.

Each area has criteria; a criterion is met when a file matching one of its patterns exists. Only file
names are read, never their contents, so "not found" means not visible in the repo, not "not done".
The criteria in config.READINESS are a draft until approved.
"""
import re

from ..common import config


def _example(patterns: list[str], paths: dict[str, str]) -> str | None:
    for lowered, original in paths.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            return original
    return None


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
    paths = {path.replace("\\", "/").lower(): path for path in sorted(set(files))}
    areas = []
    for area_id, area in settings["areas"].items():
        found, missing, earned, possible = [], [], 0, 0
        for criterion in area["criteria"].values():
            possible += criterion["points"]
            example = _example(criterion["patterns"], paths)
            if example:
                earned += criterion["points"]
                found.append({"text": criterion["text"], "file": example})
            else:
                missing.append(criterion["text"])
        areas.append({"id": area_id, "label": area["label"], "stage": area["stage"],
                      "score": round(settings["max_score"] * earned / possible),
                      "reason": _reason(found, missing), "found": found, "missing": missing})
    return {"score": round(sum(a["score"] for a in areas) / len(areas)), "areas": areas,
            "files_checked": len(paths), "status": settings["status"]}
