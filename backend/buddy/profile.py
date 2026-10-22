"""My profile: a private page about the signed-in person.

    GET /app/me/profile    about me, my teams, strengths and growth areas (each with its evidence), notes
    PUT /app/me/notes      save my own private notes (strengths, things to improve, goals)

Private by design: there is no user id in the path, so a person can only ever read their own profile.
Research-dataset teams are left out. Strengths and growth areas are worked out with simple, fixed
rules from data the app already has:
    Component 2 (Behaviour)  the person's mix of project stages, from commits, notebook cells and chat
    Component 1 (Buddy)      the planning decisions they proposed or agreed with, and the reasons they gave
    Chat                     how many messages they sent, against the team average
Every item says what it is based on, so nobody has to trust a hidden score.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from buddy.auth import current_user, get_conn, public_user
from buddy.store import now, rows_as_dicts

log = logging.getLogger("buddy")

router = APIRouter(prefix="/app", tags=["Buddy.ds app (my profile)"])

STAGE_LABEL = {
    "data_cleaning": "data cleaning",
    "eda": "exploratory analysis (EDA)",
    "feature_engineering": "feature engineering",
    "model_development": "model development",
    "model_evaluation": "model evaluation",
    "deployment_mlops": "deployment / MLOps",
    "documentation": "documentation",
}
STRENGTH_SHARE = 0.30        # a stage with at least this share of a person's work is a strength
GROWTH_SHARE = 0.05          # stages under this share are growth areas
MIN_WORK_ITEMS = 5           # fewer weighted work items than this: too little to judge the stage mix
MIN_PROPOSALS = 3            # fewer proposals than this: too few to judge how the person proposes
BEHAVIOUR = "Behaviour (Component 2)"
BUDDY = "Buddy (Component 1)"
CHAT = "Team chat"

NOTES_TABLE = """
CREATE TABLE IF NOT EXISTS user_profile_notes (
    user_id    TEXT PRIMARY KEY,
    strengths  TEXT,
    improve    TEXT,
    goals      TEXT,
    updated_at TEXT
)
"""
_notes_ready = False


def _commit(conn: Any) -> None:
    commit = getattr(conn, "commit", None)
    if commit:
        commit()


def _ensure_notes_table(conn: Any) -> None:
    global _notes_ready
    if not _notes_ready:
        conn.execute(NOTES_TABLE)
        _commit(conn)
        _notes_ready = True


def _item(title: str, evidence: str, source: str) -> dict[str, str]:
    return {"title": title, "evidence": evidence, "source": source}


def _pct(share: float) -> str:
    return f"{round(share * 100)}%"


def _joined(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + f" and {words[-1]}"


# --------------------------------------------------------------- the data
def my_teams(conn: Any, user_id: str) -> list[dict]:
    rows = rows_as_dicts(conn.execute(
        "SELECT t.team_id, t.name, m.team_role, m.member_code, m.git_identities, m.joined_at "
        "FROM team_members m JOIN teams t ON t.team_id = m.team_id "
        "WHERE m.user_id = ? AND t.is_research = 0 ORDER BY t.name", (user_id,)))
    for r in rows:
        try:
            r["git_identities"] = json.loads(r["git_identities"] or "[]")
        except (TypeError, ValueError):
            r["git_identities"] = []
    return rows


def work_mix(conn: Any, teams: list[dict]) -> Optional[dict]:
    """The person's stage mix over all their teams (Component 2), weighted like Component 2 does."""
    from component2.api.service import CHAT_WEIGHT, member_roles

    counts: Counter = Counter()
    n_code = n_chat = 0
    for t in teams:
        if not t["member_code"]:
            continue
        try:
            roles = member_roles(conn, t["team_id"], "all", roster=[t["member_code"]])
        except Exception:
            log.warning("Profile could not read the behaviour mix for team %s", t["team_id"], exc_info=True)
            continue
        me = next((m for m in (roles or {}).get("members", []) if m["member_id"] == t["member_code"]), None)
        if not me:
            continue
        weight = me["code_items"] + CHAT_WEIGHT * me["chat_items"]
        for stage, share in me["combined_mix"].items():
            counts[stage] += share * weight
        n_code += me["code_items"]
        n_chat += me["chat_items"]
    total = sum(counts.values())
    if total == 0:
        return None
    return {"mix": {s: counts[s] / total for s in STAGE_LABEL}, "weighted_items": total,
            "code_items": n_code, "chat_items": n_chat}


def decision_record(conn: Any, teams: list[dict]) -> dict[str, int]:
    """What the person did in Buddy's planning decisions, over all their teams (Component 1)."""
    from component1.common.events import EventStore
    from component1.common.schema import DecisionStatus
    from component1.decisions.tracker import is_planning_decision
    from component1.storage import load_events

    out = Counter(proposed=0, with_reason=0, adopted=0, rejected=0, agreed=0)
    for t in teams:
        code = t["member_code"]
        if not code:
            continue
        try:
            state = EventStore.from_events(t["team_id"], load_events(conn, t["team_id"])).state
        except Exception:      # Buddy has not heard this team yet (or its tables are not there)
            continue
        for d in state.decisions.values():
            if not is_planning_decision(d):
                continue
            if d.proposed_by == code:
                out["proposed"] += 1
                out["with_reason"] += 1 if d.rationale else 0
                out["adopted"] += 1 if d.status in (DecisionStatus.CONFIRMED, DecisionStatus.FINALIZED) else 0
                out["rejected"] += 1 if d.status == DecisionStatus.REJECTED else 0
            elif code in d.agreed_by:
                out["agreed"] += 1
    return dict(out)


def chat_activity(conn: Any, user_id: str, teams: list[dict]) -> dict[str, float]:
    """The person's chat messages and the average of their teammates (supervisors not counted)."""
    mine = others = teammates = 0
    for t in teams:
        members = [u for (u,) in conn.execute(
            "SELECT user_id FROM team_members WHERE team_id = ? AND team_role != 'supervisor'",
            (t["team_id"],)).fetchall()]
        counts = dict(conn.execute(
            "SELECT user_id, COUNT(*) FROM chat_messages WHERE team_id = ? GROUP BY user_id",
            (t["team_id"],)).fetchall())
        mine += counts.get(user_id, 0)
        for u in members:
            if u != user_id:
                others += counts.get(u, 0)
                teammates += 1
    return {"mine": mine, "team_average": others / teammates if teammates else 0.0}


# -------------------------------------------------------------- the rules
def strengths_and_growth(mix: Optional[dict], decisions: dict[str, int], chat: dict[str, float]):
    strengths: list[dict] = []
    growth: list[dict] = []

    if mix and mix["weighted_items"] >= MIN_WORK_ITEMS:
        work = f"{mix['code_items']} code items and {mix['chat_items']} chat messages"
        for stage, share in sorted(mix["mix"].items(), key=lambda kv: -kv[1]):
            if share >= STRENGTH_SHARE:
                strengths.append(_item(f"Strong in {STAGE_LABEL[stage]}",
                                       f"{_pct(share)} of your work ({work}).", BEHAVIOUR))
        little = [STAGE_LABEL[s] for s, share in mix["mix"].items() if share < GROWTH_SHARE]
        if little:
            growth.append(_item(f"Little activity in {_joined(little)}",
                                f"Each is under {_pct(GROWTH_SHARE)} of your work so far.", BEHAVIOUR))

    n = decisions.get("proposed", 0)
    if n >= MIN_PROPOSALS:
        strengths.append(_item("Drives planning decisions", f"You proposed {n} planning decisions.", BUDDY))
    if n >= 2 and decisions["with_reason"] / n >= 0.5:
        strengths.append(_item("Explains choices",
                               f"You gave a reason for {decisions['with_reason']} of {n} proposals.", BUDDY))
    if n >= 2 and decisions["adopted"] / n >= 0.6:
        strengths.append(_item("Ideas the team adopts",
                               f"{decisions['adopted']} of {n} of your proposals were agreed or finalized.", BUDDY))
    if decisions.get("agreed", 0) >= 3:
        strengths.append(_item("Backs teammates' proposals",
                               f"You agreed with {decisions['agreed']} of your teammates' proposals.", BUDDY))
    if n >= MIN_PROPOSALS and decisions["with_reason"] / n < 0.34:
        growth.append(_item("Give reasons when proposing",
                            f"{decisions['with_reason']} of {n} proposals had a reason.", BUDDY))
    if n >= MIN_PROPOSALS and decisions["rejected"] / n >= 0.5:
        growth.append(_item("Check ideas with the team before proposing",
                            f"{decisions['rejected']} of {n} proposals were rejected.", BUDDY))

    mine, avg = chat["mine"], chat["team_average"]
    if mine >= 10 and avg > 0 and mine >= 1.5 * avg:
        strengths.append(_item("Active in the team chat",
                               f"You sent {mine} messages; your teammates average {avg:.0f}.", CHAT))
    if avg >= 4 and mine <= 0.5 * avg:
        growth.append(_item("Speak up more in the team chat",
                            f"You sent {mine} messages; your teammates average {avg:.0f}.", CHAT))
    return strengths, growth


# ---------------------------------------------------------------- notes
def read_notes(conn: Any, user_id: str) -> dict:
    _ensure_notes_table(conn)
    row = conn.execute("SELECT strengths, improve, goals, updated_at FROM user_profile_notes WHERE user_id = ?",
                       (user_id,)).fetchone()
    keys = ("strengths", "improve", "goals", "updated_at")
    return dict(zip(keys, row)) if row else dict.fromkeys(keys, None)


def save_notes(conn: Any, user_id: str, strengths: str, improve: str, goals: str) -> dict:
    _ensure_notes_table(conn)
    conn.execute(
        "INSERT INTO user_profile_notes (user_id, strengths, improve, goals, updated_at) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT (user_id) DO UPDATE SET strengths = excluded.strengths, improve = excluded.improve, "
        "goals = excluded.goals, updated_at = excluded.updated_at",
        (user_id, strengths, improve, goals, now()))
    _commit(conn)
    return read_notes(conn, user_id)


# --------------------------------------------------------------- profile
def build_profile(conn: Any, user: dict) -> dict:
    teams = my_teams(conn, user["user_id"])
    student_teams = [t for t in teams if t["team_role"] in ("lead", "member")]
    profile = {
        "about": {**public_user(user), "joined": user.get("created_at")},
        "teams": [{k: t[k] for k in ("team_id", "name", "team_role", "member_code", "git_identities", "joined_at")}
                  for t in teams],
        "notes": read_notes(conn, user["user_id"]),
        "has_activity": False,
        "work_mix": None,
        "strengths": [],
        "growth": [],
    }
    if not student_teams:
        return profile                      # supervisors and examiners: about me, groups and notes only
    mix = work_mix(conn, student_teams)
    decisions = decision_record(conn, student_teams)
    chat = chat_activity(conn, user["user_id"], student_teams)
    strengths, growth = strengths_and_growth(mix, decisions, chat)
    profile.update(
        has_activity=bool(mix) or decisions["proposed"] + decisions["agreed"] > 0 or chat["mine"] > 0,
        work_mix=mix,
        decisions=decisions,
        chat=chat,
        strengths=strengths,
        growth=growth,
    )
    return profile


class NotesIn(BaseModel):
    strengths: str = Field("", max_length=2000)
    improve: str = Field("", max_length=2000)
    goals: str = Field("", max_length=2000)


@router.get("/me/profile")
def get_profile(user=Depends(current_user), conn=Depends(get_conn)):
    return build_profile(conn, user)


@router.put("/me/notes")
def put_notes(body: NotesIn, user=Depends(current_user), conn=Depends(get_conn)):
    return save_notes(conn, user["user_id"], body.strengths.strip(), body.improve.strip(), body.goals.strip())
