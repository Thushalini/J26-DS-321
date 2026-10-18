"""
Step 5a: Builds Component 2's results as plain Python dicts (later turned into JSON by the API).

Kept separate from the web code (router.py) so it can be tested and reused
without running a server.
"""
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from component2.config import STAGES

LAGGING_SHARE = 0.05          # a stage with < 5% of recent work is "lagging" (agree with C4)
TREND_WEEKS = 12              # how many recent weeks of role history to return
SEVERITY_ORDER = ["none", "minor", "moderate", "significant"]


def _parse(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_teams(conn):
    rows = conn.execute("SELECT team_id, COUNT(*), MIN(ts), MAX(ts) FROM activities "
                        "GROUP BY team_id ORDER BY team_id").fetchall()
    return [{"team_id": t, "activities": n, "first_activity": first, "last_activity": last}
            for t, n, first, last in rows]


def _labels_for_team(conn, team_id):
    """activity_id -> stage, preferring the trained model over the keyword baseline."""
    labels = {}
    for labeller in ("baseline", "model"):
        rows = conn.execute("SELECT l.activity_id, l.stage FROM stage_labels l "
                            "JOIN activities a ON a.activity_id = l.activity_id "
                            "WHERE a.team_id = ? AND l.labeller = ?", (team_id, labeller)).fetchall()
        labels.update(dict(rows))
    return labels


def stage_breakdown(conn, team_id, days):
    """Share of the team's recent work (commits + notebook cells) in each stage."""
    rows = conn.execute("SELECT activity_id, ts FROM activities WHERE team_id = ? AND source != 'chat' "
                        "ORDER BY ts", (team_id,)).fetchall()
    if not rows:
        return None, {}
    end = _parse(rows[-1][1])
    start = end - timedelta(days=days)
    labels = _labels_for_team(conn, team_id)
    counts = Counter(labels.get(aid) for aid, ts in rows if _parse(ts) >= start)
    total = sum(counts[s] for s in STAGES)
    shares = {s: round(counts[s] / total, 3) if total else 0.0 for s in STAGES}
    window = {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": rows[-1][1]}
    return window, shares


def daily_stage_counts(conn, team_id, days):
    """For C3: number of activities per stage per day in the recent window."""
    window, _ = stage_breakdown(conn, team_id, days)
    if window is None:
        return []
    labels = _labels_for_team(conn, team_id)
    rows = conn.execute("SELECT activity_id, ts FROM activities WHERE team_id = ? AND source != 'chat' "
                        "AND ts >= ? ORDER BY ts", (team_id, window["start"])).fetchall()
    per_day = {}
    for aid, ts in rows:
        day = ts[:10]
        per_day.setdefault(day, Counter())[labels.get(aid, "other")] += 1
    return [{"date": day, **{s: c[s] for s in STAGES + ["other"]}} for day, c in sorted(per_day.items())]


def drift(conn, team_id):
    """Latest drift result per dimension, plus the worst severity overall."""
    rows = conn.execute("SELECT window_end, dimension, planned, seen_top, drift_score, severity, explanation "
                        "FROM drift_results WHERE team_id = ? ORDER BY window_end", (team_id,)).fetchall()
    latest = {}
    for window_end, dim, planned, seen_top, score, severity, why in rows:
        latest[dim] = {"dimension": dim, "planned": planned, "seen_top": seen_top,
                       "drift_score": score, "severity": severity, "explanation": why,
                       "window_end": window_end}
    graded = [d["severity"] for d in latest.values() if d["severity"] in SEVERITY_ORDER]
    overall = max(graded, key=SEVERITY_ORDER.index) if graded else "unknown"
    return {"overall_severity": overall, "dimensions": list(latest.values())}


def members(conn, team_id):
    """Each member's latest role vector and recent role history."""
    rows = conn.execute("SELECT member_id, week, stage_mix, chat, dominant_stage FROM role_trends "
                        "WHERE team_id = ? ORDER BY member_id, week", (team_id,)).fetchall()
    result = {}
    for member_id, week, mix, chat, dominant in rows:
        m = result.setdefault(member_id, {"member_id": member_id, "role_trend": []})
        m["stage_mix"] = json.loads(mix or "{}")
        m["chat"] = json.loads(chat or "{}")
        m["dominant_role"] = dominant
        m["role_trend"].append({"week": week, "dominant": dominant})
    for m in result.values():
        m["role_trend"] = m["role_trend"][-TREND_WEEKS:]
    return list(result.values())


def team_summary(conn, team_id, days=30):
    """The full Component 2 output for one team (the JSON agreed with C3 and C4)."""
    window, shares = stage_breakdown(conn, team_id, days)
    if window is None:
        return None
    lagging = [s for s, share in shares.items() if share < LAGGING_SHARE]
    return {
        "component": "C2",
        "version": "1.0",
        "team_id": team_id,
        "generated_at": _now(),
        "window": window,
        "stage_breakdown": shares,
        "lagging_stages": lagging,
        "drift": drift(conn, team_id),
        "members": members(conn, team_id),
    }

# ---------- Role snapshots over different time windows ----------

WINDOWS = {"3d": 3, "7d": 7, "30d": 30, "all": None}
CHAT_WEIGHT = 0.5      # a chat message about a stage counts half as much as a commit / notebook edit
MAIN_ROLE_SHARE = 0.25  # stages with >= 25% of a member's work are listed as their main roles


def _shares(counter):
    total = sum(counter[s] for s in STAGES)
    return {s: round(counter[s] / total, 3) if total else 0.0 for s in STAGES}


def member_roles(conn, team_id, window="7d"):
    """Every member's mix of ALL stages in one window (3 days, 7 days, 30 days or the whole project),
    using code (commits + notebook cells) AND chat. Chat is shown separately and counts at half weight,
    because talking about a stage is weaker evidence than doing it."""
    rows = conn.execute("SELECT activity_id, member_id, source, ts FROM activities WHERE team_id = ? "
                        "ORDER BY ts", (team_id,)).fetchall()
    if not rows:
        return None
    end = _parse(rows[-1][3])
    days = WINDOWS[window]
    start = end - timedelta(days=days) if days else _parse(rows[0][3])
    labels = _labels_for_team(conn, team_id)

    code, chat = defaultdict(Counter), defaultdict(Counter)
    members = sorted({m for _, m, _, _ in rows if m})
    for aid, member, source, ts in rows:
        if not member or _parse(ts) < start:
            continue
        stage = labels.get(aid)
        if stage in STAGES:
            (chat if source == "chat" else code)[member][stage] += 1

    result = []
    for m in members:
        combined = Counter()
        for s in STAGES:
            combined[s] = code[m][s] + CHAT_WEIGHT * chat[m][s]
        mix = _shares(combined)
        ranked = sorted((s for s in STAGES if mix[s] > 0), key=mix.get, reverse=True)
        main = [s for s in ranked if mix[s] >= MAIN_ROLE_SHARE] or ranked[:1]
        n_code, n_chat = sum(code[m].values()), sum(chat[m].values())
        evidence = "none" if n_code + n_chat == 0 else ("low" if n_code + n_chat < 3 else "ok")
        result.append({"member_id": m, "code_items": n_code, "chat_items": n_chat,
                       "combined_mix": mix, "code_mix": _shares(code[m]), "chat_mix": _shares(chat[m]),
                       "main_roles": main, "evidence": evidence})
    return {"team_id": team_id, "window": window,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": rows[-1][3],
            "chat_weight": CHAT_WEIGHT, "members": result}