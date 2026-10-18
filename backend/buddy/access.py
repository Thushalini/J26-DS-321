"""Who can see / change which team.

  Real team      -> its members (lead, member) and the supervisors who joined it.
  Research team  -> supervisors and examiners only (read-only), labelled as the
                    Phase 1 research dataset. Nobody is a "member" of a research team,
                    so without this rule they would disappear once sign-in exists.
  Real team, examiner -> read-only and ANONYMISED: behaviour output by member code only;
                    no names, emails, git identities, repo link, plan notes or chat.
"""
import hashlib

from fastapi import HTTPException

from buddy.auth import STAFF_ROLES
from buddy.store import one_as_dict, rows_as_dicts


def membership(conn, team_id, user_id):
    return one_as_dict(conn.execute("SELECT * FROM team_members WHERE team_id = ? AND user_id = ?",
                                    (team_id, user_id)))


def visible_teams(conn, user):
    mine = rows_as_dicts(conn.execute(
        "SELECT t.*, m.team_role AS my_role FROM teams t JOIN team_members m ON m.team_id = t.team_id "
        "WHERE m.user_id = ? ORDER BY t.name", (user["user_id"],)))
    if user["role"] not in STAFF_ROLES:
        return mine
    research = rows_as_dicts(conn.execute(
        "SELECT t.*, 'viewer' AS my_role FROM teams t WHERE t.is_research = 1 ORDER BY t.name"))
    if user["role"] != "examiner":
        return mine + research
    live = rows_as_dicts(conn.execute(
        "SELECT t.*, 'viewer' AS my_role FROM teams t WHERE t.is_research = 0 ORDER BY t.created_at"))
    return live + research


def team_access(conn, user, team_id):
    """Returns (team, my_role) or raises 404/403. my_role: lead | member | supervisor | viewer."""
    team = one_as_dict(conn.execute("SELECT * FROM teams WHERE team_id = ?", (team_id,)))
    if not team:
        raise HTTPException(404, f"Team '{team_id}' not found.")
    m = membership(conn, team_id, user["user_id"])
    if m:
        return team, m["team_role"]
    if team["is_research"] and user["role"] in STAFF_ROLES:
        return team, "viewer"
    if user["role"] == "examiner":
        return team, "viewer"                     # real team: read-only and anonymised (see is_anonymised)
    raise HTTPException(403, "You are not a member of this team.")


def is_anonymised(team, my_role):
    """A real team seen by an examiner: hide everything that identifies a person."""
    return my_role == "viewer" and not team["is_research"]


def anonymous_team_name(team_id):
    """Stable stand-in for a real team's name (the name itself may identify students)."""
    return f"Team {hashlib.sha256(team_id.encode()).hexdigest()[:4].upper()}"


def require(role, allowed, action):
    if role not in allowed:
        raise HTTPException(403, f"Only {' / '.join(allowed)} can {action}.")


def require_live(team, action):
    if team["is_research"]:
        raise HTTPException(400, f"Research dataset teams are read-only - you can't {action}.")
