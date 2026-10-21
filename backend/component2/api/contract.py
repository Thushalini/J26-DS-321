"""
Component 2's interface contract: what it accepts, what it returns, and what counts as missing,
invalid or delayed. The full description (owners, formats, fallbacks) is in docs/interfaces.md.

Inputs are checked with check_activity() and clean_plan() before they are used.
Outputs can be checked with check_summary(); the tests run it on every summary C2 builds,
and C3 / C4 can call it on what they receive.
"""
import re
from datetime import datetime, timedelta, timezone

from component2.config import STAGES

CONTRACT_VERSION = "1.1"        # 1.1 added data_status (1.0 fields are unchanged)

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
SOURCES = {"commit", "notebook_cell", "code_change", "chat"}
DIMENSIONS = ("target", "algorithm", "metric")
SEVERITIES = {"none", "minor", "moderate", "significant", "no_plan", "no_evidence"}
OVERALL_SEVERITIES = {"none", "minor", "moderate", "significant", "unknown"}

STALE_AFTER_DAYS = 14           # no new activity for 2 weeks -> the summary is marked "stale"
FUTURE_TOLERANCE = timedelta(days=1)   # clocks in different time zones; more than this is invalid
MAX_PLAN_LENGTH = 200


def parse_ts(ts):
    """The one timestamp format C2 uses everywhere (UTC). None if it doesn't match."""
    try:
        return datetime.strptime(ts, TS_FORMAT)
    except (TypeError, ValueError):
        return None


def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------- Inputs ----------

def check_activity(row, now=None):
    """One activity (commit, notebook_cell, code_change or chat message) before it is saved.
    Returns None if it is fine, otherwise the reason it is rejected."""
    for field in ("activity_id", "team_id", "source", "ts"):
        if not str(row.get(field) or "").strip():
            return f"missing {field}"
    if row["source"] not in SOURCES:
        return f"unknown source '{row['source']}'"
    ts = parse_ts(row["ts"])
    if ts is None:
        return f"timestamp '{row['ts']}' is not in {TS_FORMAT} form"
    if ts > (now or utc_now()) + FUTURE_TOLERANCE:
        return f"timestamp {row['ts']} is in the future"
    if not str(row.get("text") or "").strip():
        return "empty text"
    return None


def split_valid(rows, now=None):
    """(rows to keep, [(activity_id, reason), ...] rejected)."""
    keep, rejected = [], []
    for r in rows:
        reason = check_activity(r, now)
        if reason:
            rejected.append((r.get("activity_id"), reason))
        else:
            keep.append(r)
    return keep, rejected


C1_FIELDS = {"target": "target_variable", "algorithm": "algorithm", "metric": "evaluation_metric"}
C1_SETTLED = {"known", "confirmed"}     # C1 fields still missing / uncertain / conflicting are not a plan yet


def clean_plan(blueprint):
    """Blueprint from C1 / the app -> {dimension: value or None}.
    Accepts the app's {"target": "churn"}, the blueprint files' {"target": {"name": "churn"}} and
    C1's Blueprint JSON {"fields": {"target_variable": {"value": "churn", "status": "known"}}}.
    Anything that is not a short non-empty text becomes None, which C2 reports as severity
    'no_plan' (never guessed)."""
    blueprint = blueprint if isinstance(blueprint, dict) else {}
    c1_fields = blueprint.get("fields") if isinstance(blueprint.get("fields"), dict) else None
    plan = {}
    for d in DIMENSIONS:
        if c1_fields is not None:
            field = c1_fields.get(C1_FIELDS[d]) or {}
            value = field.get("value") if field.get("status") in C1_SETTLED else None
        else:
            value = blueprint.get(d)
        if isinstance(value, dict):
            value = value.get("name")
        value = value.strip() if isinstance(value, str) else None
        plan[d] = value if value and len(value) <= MAX_PLAN_LENGTH else None
    return plan


# ---------- Outputs ----------

def data_status(last_activity, missing, now=None):
    """How complete and how recent a team's results are, so C3 / C4 can decide how far to trust them.
      ok          everything was computed and the latest activity is recent
      incomplete  some parts were not computed yet (listed in `missing`)
      stale       complete, but no new activity for STALE_AFTER_DAYS days (sync late or team quiet)"""
    last = parse_ts(last_activity)
    age = (now or utc_now()) - last if last else None
    days = max(0, age.days) if age is not None else None
    if missing:
        status = "incomplete"
    elif days is None or days > STALE_AFTER_DAYS:
        status = "stale"
    else:
        status = "ok"
    return {"status": status, "last_activity": last_activity, "days_since_last_activity": days,
            "stale_after_days": STALE_AFTER_DAYS, "missing": sorted(missing)}


def _share(value):
    return isinstance(value, (int, float)) and 0 <= value <= 1


def check_summary(payload):
    """Problems with a /c2/teams/{id}/summary payload ([] = it follows the contract)."""
    if not isinstance(payload, dict):
        return ["summary is not an object"]
    problems = []
    for key in ("component", "version", "team_id", "generated_at", "window", "stage_breakdown",
                "lagging_stages", "drift", "members", "data_status"):
        if key not in payload:
            problems.append(f"missing '{key}'")
    if problems:
        return problems

    if parse_ts(payload["generated_at"]) is None:
        problems.append("generated_at is not a UTC timestamp")
    window = payload["window"] or {}
    if parse_ts(window.get("start")) is None or parse_ts(window.get("end")) is None:
        problems.append("window start / end are not UTC timestamps")

    shares = payload["stage_breakdown"]
    if set(shares) != set(STAGES):
        problems.append(f"stage_breakdown must have exactly the stages {STAGES}")
    elif not all(_share(v) for v in shares.values()):
        problems.append("stage_breakdown values must be between 0 and 1")
    if not set(payload["lagging_stages"]) <= set(STAGES):
        problems.append("lagging_stages contains an unknown stage")

    drift = payload["drift"] or {}
    if drift.get("overall_severity") not in OVERALL_SEVERITIES:
        problems.append(f"drift.overall_severity '{drift.get('overall_severity')}' is not allowed")
    for d in drift.get("dimensions", []):
        if d.get("dimension") not in DIMENSIONS:
            problems.append(f"unknown drift dimension '{d.get('dimension')}'")
        if d.get("severity") not in SEVERITIES:
            problems.append(f"drift severity '{d.get('severity')}' is not allowed")
        if d.get("drift_score") is not None and not _share(d["drift_score"]):
            problems.append(f"drift_score for '{d.get('dimension')}' must be between 0 and 1 or null")

    for m in payload["members"]:
        if not re.fullmatch(r"M-[0-9a-f]{6}", str(m.get("member_id"))):
            problems.append(f"member_id '{m.get('member_id')}' is not an anonymised code")

    if payload["data_status"].get("status") not in {"ok", "incomplete", "stale"}:
        problems.append("data_status.status must be ok, incomplete or stale")
    return problems
