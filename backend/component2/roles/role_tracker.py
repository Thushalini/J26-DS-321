"""
Step 4c: Dynamic role tracker (FR07, objective O4).

For every member, every week, we build one vector with two parts:

  WHAT they work on (from commits + notebook cells):
      share of their work in each of the 7 lifecycle stages
  HOW they take part in team talk (from chat, GCA-style, Dowell et al. 2019):
      participation  = their share of all team messages
      responsiveness = how much their replies relate to what others said
      social_impact  = how much others reply to them, and how related those replies are
      ("how related" = MiniLM cosine similarity between a reply and the message it answers)

The weekly vectors are smoothed with an exponential moving average (EMA)
so one unusual week doesn't flip someone's role. The member's role each week
is the stage with the biggest smoothed share. A ROLE SHIFT is reported when
that role changes and stays changed for at least 2 weeks.

Run from backend:
    python -m component2.roles.role_tracker
"""
import json
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from component2.config import STAGES
from component2.db import connect, save_roles
from component2.embedder import embed
from component2.research_teams import research_team_ids

ALPHA = 0.5            # EMA weight of the newest week (0.5 = this week and the past count equally)
MIN_WEEKS_FOR_SHIFT = 2
CHAT_KEYS = ("participation", "responsiveness", "social_impact")


def iso_week(ts):
    y, w, _ = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").isocalendar()
    return f"{y}-W{w:02d}"


def stage_labels(conn):
    """Model labels if they exist, otherwise keyword-baseline labels."""
    labels = {}
    for labeller in ("baseline", "model"):
        for aid, stage in conn.execute("SELECT activity_id, stage FROM stage_labels WHERE labeller = ?",
                                       (labeller,)).fetchall():
            labels[aid] = stage
    return labels


def weekly_stage_mix(work_rows, labels, members):
    """work_rows: (activity_id, member_id, week) -> {week: {member: {stage: share}}}"""
    counts = defaultdict(lambda: defaultdict(Counter))
    for aid, member, week in work_rows:
        stage = labels.get(aid, "other")
        if stage in STAGES:
            counts[week][member][stage] += 1
    mix = {}
    for week, per_member in counts.items():
        mix[week] = {}
        for m in members:
            total = sum(per_member[m].values())
            mix[week][m] = {s: (per_member[m][s] / total if total else 0.0) for s in STAGES}
    return mix


def weekly_chat_measures(chat_rows, members):
    """chat_rows: (activity_id, member_id, week, text, reply_to) -> {week: {member: {measure: value}}}"""
    if not chat_rows:
        return {}
    vectors = embed([r[3] for r in chat_rows])
    index = {r[0]: n for n, r in enumerate(chat_rows)}
    author = {r[0]: r[1] for r in chat_rows}

    by_week = defaultdict(list)
    for r in chat_rows:
        by_week[r[2]].append(r)

    result = {}
    for week, rows in by_week.items():
        sent = Counter(r[1] for r in rows)
        responsive = Counter()
        impact = Counter()
        for aid, member, _, _, parent in rows:
            if parent and parent in index and author[parent] != member:
                sim = float(vectors[index[aid]] @ vectors[index[parent]])
                responsive[member] += sim                 # I answered someone, this related
                impact[author[parent]] += sim             # someone answered them, this related
        total = sum(sent.values())
        result[week] = {m: {"participation": sent[m] / total,
                            "responsiveness": responsive[m] / sent[m] if sent[m] else 0.0,
                            "social_impact": impact[m] / sent[m] if sent[m] else 0.0}
                        for m in members}
    return result


def ema(previous, current):
    if previous is None:
        return dict(current)
    return {k: ALPHA * current[k] + (1 - ALPHA) * previous[k] for k in current}


def find_shifts(weekly_roles):
    """weekly_roles: list of (week, role). A shift = new role that lasts >= MIN_WEEKS_FOR_SHIFT weeks."""
    shifts, current = [], None
    for i, (week, role) in enumerate(weekly_roles):
        if role is None or role == current:
            continue
        upcoming = [r for _, r in weekly_roles[i:i + MIN_WEEKS_FOR_SHIFT]]
        if len(upcoming) == MIN_WEEKS_FOR_SHIFT and all(r == role for r in upcoming):
            if current is not None:
                shifts.append((week, current, role))
            current = role
    return shifts


def track_team(conn, team_id, labels):
    rows = conn.execute("SELECT activity_id, member_id, source, ts, text, meta FROM activities "
                        "WHERE team_id = ? ORDER BY ts", (team_id,)).fetchall()
    members = sorted({r[1] for r in rows if r[1]})
    work = [(r[0], r[1], iso_week(r[3])) for r in rows if r[2] != "chat"]
    chat = [(r[0], r[1], iso_week(r[3]), r[4], json.loads(r[5] or "{}").get("reply_to"))
            for r in rows if r[2] == "chat"]

    mix = weekly_stage_mix(work, labels, members)
    talk = weekly_chat_measures(chat, members)
    weeks = sorted(set(mix) | set(talk))
    empty_mix = {s: 0.0 for s in STAGES}
    empty_talk = {k: 0.0 for k in CHAT_KEYS}

    out, smooth_mix, smooth_talk = [], {}, {}
    for week in weeks:
        for m in members:
            this_mix = mix.get(week, {}).get(m)
            if this_mix and sum(this_mix.values()) > 0:          # weeks without work keep the old mix
                smooth_mix[m] = ema(smooth_mix.get(m), this_mix)
            smooth_talk[m] = ema(smooth_talk.get(m), talk.get(week, {}).get(m, empty_talk))
            s_mix = smooth_mix.get(m, empty_mix)
            dominant = max(s_mix, key=s_mix.get) if sum(s_mix.values()) > 0 else None
            out.append({"team_id": team_id, "member_id": m, "week": week,
                        "stage_mix": {k: round(v, 3) for k, v in s_mix.items()},
                        "chat": {k: round(v, 3) for k, v in smooth_talk[m].items()},
                        "dominant_stage": dominant})
    return out


def main():
    conn = connect()
    labels = stage_labels(conn)
    teams = research_team_ids(conn)          # real teams are tracked by buddy.sync
    for team in teams:
        rows = track_team(conn, team, labels)
        save_roles(conn, rows)
        print(f"\n{team}")
        per_member = defaultdict(list)
        for r in rows:
            per_member[r["member_id"]].append(r)
        for m, mrows in per_member.items():
            roles = [(r["week"], r["dominant_stage"]) for r in mrows]
            shifts = find_shifts(roles)
            last = mrows[-1]
            avg_part = np.mean([r["chat"]["participation"] for r in mrows])
            shift_txt = "; ".join(f"{w}: {a} -> {b}" for w, a, b in shifts) or "no lasting shift"
            print(f"  {m}  final role: {last['dominant_stage'] or '-':<20} "
                  f"chat participation {avg_part:.2f}  | {shift_txt}")


if __name__ == "__main__":
    main()