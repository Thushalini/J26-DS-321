"""Module 4: MLOps readiness, scored from the group's repository.

Each area has criteria. With a repo scan (repo_scan.py), criteria that have a content rule are judged on what
the key files contain (e.g. at least 3 tests, CI that runs them); the rest, and every criterion when GitHub can't
be read, are judged on file names: "not found" then means not visible in the repo, not "not done".
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


def _content_ok(rule: dict, value) -> bool:
    if value is None:
        return False
    return value == rule["is"] if "is" in rule else value >= rule["at_least"]


def check(criteria: dict, paths: dict[str, str], evidence: dict | None = None) -> dict:
    found, missing, earned, possible = [], [], 0, 0
    for criterion in criteria.values():
        content = criterion.get("content")
        if evidence is None and content and not criterion["patterns"]:
            continue
        possible += criterion["points"]
        example = first_match(criterion["patterns"], paths) if criterion["patterns"] else None
        label = criterion["text"]
        if evidence is not None and content:
            value = evidence.get(content["evidence"])
            met = _content_ok(content, value)
            if content.get("detail") and value is not None:
                label = f"{label} ({content['detail'].format(value=value)})"
        else:
            met = example is not None
        if not met and evidence and evidence.get(criterion.get("or_evidence", "")):
            met = True
        if met:
            earned += criterion["points"]
            found.append({"text": label, "file": example})
        else:
            missing.append(label)
    return {"share": earned / possible if possible else 0.0, "found": found, "missing": missing}


def _reason(found: list[dict], missing: list[str]) -> str:
    text = config.READINESS["text"]
    parts = []
    if found:
        parts.append(text["found"].format(items=", ".join(item["text"] for item in found)))
    if missing:
        parts.append(text["missing"].format(items=", ".join(missing)))
    return " ".join(parts)


def assess(files: list[str], evidence: dict | None = None) -> dict | None:
    if not files:
        return None
    settings = config.READINESS
    paths = normalise(files)
    areas = []
    for area_id, area in settings["areas"].items():
        result = check(area["criteria"], paths, evidence)
        areas.append({"id": area_id, "label": area["label"], "stage": area["stage"],
                      "score": round(settings["max_score"] * result["share"]),
                      "reason": _reason(result["found"], result["missing"]),
                      "found": result["found"], "missing": result["missing"]})
    return {"score": round(sum(a["score"] for a in areas) / len(areas)), "areas": areas,
            "files_checked": len(paths), "basis": "contents" if evidence is not None else "names",
            "status": settings["status"]}
