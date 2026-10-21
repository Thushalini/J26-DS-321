"""Component 2 adapter: behaviour summary and member roles -> drift and contribution-balance features.

Agreed with Thushalini on 2026-10-18: the overall drift is the highest of the target / algorithm /
metric drift scores (she reports which one drove it); the 5% lagging-stage limit is provisional;
members with no activity are still listed (evidence "none", zero items).
Contribution balance = 1 - Gini coefficient of members' activity (1 = perfectly even, 0 = one person
does everything). Chat counts at Component 2's own chat weight.
"""
from ..common import config


def gini(values: list[float]) -> float:
    ordered = sorted(values)
    n, total = len(ordered), sum(ordered)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(ordered))
    return (2 * weighted) / (n * total) - (n + 1) / n


def overall_drift(summary: dict) -> tuple[float | None, str | None]:
    wanted = config.C2["drift_dimensions"]
    scored = [(d["drift_score"], d["dimension"]) for d in (summary.get("drift") or {}).get("dimensions", [])
              if d.get("dimension") in wanted and d.get("drift_score") is not None]
    if not scored:
        return None, None
    score, driver = max(scored)
    return float(score), driver


def contribution_balance(roles: dict | None) -> float | None:
    members = (roles or {}).get("members") or []
    if not members:
        return None
    chat_weight = roles.get("chat_weight", config.C2["default_chat_weight"])
    activity = [m.get("code_items", 0) + chat_weight * m.get("chat_items", 0) for m in members]
    if sum(activity) == 0:
        return None
    return round(1 - gini(activity), 4)


def project_features(summary: dict, roles: dict | None = None) -> dict:
    drift, driver = overall_drift(summary)
    inactive = [m["member_id"] for m in (roles or {}).get("members", [])
                if m.get("evidence") == config.C2["inactive_evidence"]]
    return {
        "features": {"drift_score": drift, "contribution_balance": contribution_balance(roles)},
        "context": {
            "drift_driver": driver,
            "drift_severity": (summary.get("drift") or {}).get("overall_severity"),
            "lagging_stages": summary.get("lagging_stages", []),
            "stage_breakdown": summary.get("stage_breakdown", {}),
            "inactive_members": inactive,
        },
    }
