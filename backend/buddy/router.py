"""Buddy.ds app API (Phase 2). All addresses start with /app and need a signed-in user
(except register / login). Try them at http://127.0.0.1:8000/docs.

  POST /app/auth/register | /app/auth/login | /app/auth/logout     GET /app/auth/me
  POST /app/auth/supervisor-code         supervisor: make a new personal code (teams already added stay)
  POST /app/auth/forgot | /app/auth/reset   email a reset code, then set a new password with it
  POST /app/auth/change-password           signed in: current password -> new password
  GET  /app/teams                        teams I can see (+ drift headline, for the supervisor overview)
  POST /app/teams                        create a real team (students; the creator becomes the lead)
  POST /app/teams/join                   join with a join code (students)
  POST /app/teams/{id}/supervisors       add a supervisor with that supervisor's code (lead, member)
  DELETE /app/teams/{id}/members/{uid}   remove a member (lead)
  PUT  /app/teams/{id}/supervision       supervisor's own status dot / remove from my list (supervisor)
  GET|POST|DELETE /app/panel-review      supervisor: look at one other team with its join code, temporarily
  GET  /app/teams/{id}                   team details, members, unlinked git authors
  PUT  /app/teams/{id}                   rename / change repo link (lead, supervisor)
  PUT  /app/teams/{id}/me                my git names / emails (to link my commits)
  POST /app/teams/{id}/link              lead / supervisor: credit an unlinked git author to a member
  POST /app/teams/{id}/sync  GET .../sync   run / watch the live Component 2 pipeline
  GET|PUT /app/teams/{id}/blueprint      planned target / algorithm / metric (stand-in for Component 1)
  GET|POST /app/teams/{id}/chat          team chat: lead + members only (research teams: the simulated chat, read-only)
  GET  /app/teams/{id}/behavior/summary  Component 2 summary, after the access check
  GET  /app/teams/{id}/behavior/roles    Component 2 role windows, after the access check
  POST /app/feedback   GET /app/feedback   GET /app/feedback.csv
"""
import csv
import io
import json
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from component2.api import service
from component2.classify.keyword_baseline import classify
from component2.config import BLUEPRINT_DIR
from component2.db import save_activities, save_labels
try:
    from component1.hook import merge_chat, run_in_background
except Exception as e:                      # Component 1 must never stop the app from starting
    print(f"[component1] not loaded: {e}")

    def run_in_background(*args, **kwargs):
        pass

    def merge_chat(conn, team_id, messages, limit=200):
        return messages

from buddy import codes, sync
from buddy.access import (PANEL_ROLE, anonymous_team_name, is_anonymised, membership, require, require_live,
                          team_access, visible_teams)
from buddy.auth import (ROLES, check_password, check_reset, current_token, current_user, end_other_sessions,
                        end_session, get_conn, hash_password, public_user, send_reset_code, staff_code_ok,
                        start_reset, start_session)
from buddy.store import TS_FORMAT, new_id, now, one_as_dict, rows_as_dicts

router = APIRouter(prefix="/app", tags=["Buddy.ds app (accounts, teams, chat, feedback)"])

RESEARCH_LABEL = "Research dataset (public repos, simulated chat)"
SUPERVISION_STATUSES = ("attention", "fine", "good")
PANEL_REVIEW_HOURS = 3


# ---------- Accounts ----------

class Register(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=8, max_length=200)
    role: str = "student"
    staff_code: str | None = None


class Login(BaseModel):
    email: str
    password: str


@router.post("/auth/register")
def register(body: Register, conn=Depends(get_conn)):
    email = body.email.strip().lower()
    if "@" not in email:
        raise HTTPException(400, "Please enter a valid email address.")
    if body.role not in ROLES:
        raise HTTPException(400, f"Role must be one of {', '.join(ROLES)}.")
    # Supervisors only see teams that gave them a join code, so they register freely.
    # Examiners read ALL feedback, so they need the STAFF_CODE from the project owner.
    if body.role == "examiner" and not staff_code_ok(body.staff_code):
        raise HTTPException(403, "Examiner accounts need the staff code from the project owner.")
    if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
        raise HTTPException(409, "An account with this email already exists. Please sign in.")
    user = {"user_id": new_id("u"), "email": email, "name": body.name.strip(), "role": body.role}
    supervisor_code = codes.new_code(conn) if body.role == "supervisor" else None
    conn.execute("INSERT INTO users (user_id, email, name, role, password_hash, created_at, supervisor_code) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (user["user_id"], email, user["name"], body.role, hash_password(body.password), now(), supervisor_code))
    conn.commit()
    if supervisor_code:
        user["supervisor_code"] = supervisor_code
    return {"token": start_session(conn, user["user_id"]), "user": user}


@router.post("/auth/login")
def login(body: Login, conn=Depends(get_conn)):
    user = one_as_dict(conn.execute("SELECT * FROM users WHERE email = ?", (body.email.strip().lower(),)))
    if not user or not check_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Wrong email or password.")
    return {"token": start_session(conn, user["user_id"]), "user": public_user(user)}


class Forgot(BaseModel):
    email: str


class Reset(BaseModel):
    email: str
    code: str
    new_password: str = Field(min_length=8, max_length=200)


class ChangePassword(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=200)


@router.post("/auth/forgot")
def forgot_password(body: Forgot, conn=Depends(get_conn)):
    # Same answer whether or not the account exists, so this can't be used to probe emails.
    user = one_as_dict(conn.execute("SELECT * FROM users WHERE email = ?", (body.email.strip().lower(),)))
    if user:
        code = start_reset(conn, user["user_id"])
        try:
            send_reset_code(user["email"], user["name"], code)
        except Exception as e:
            print(f"[buddy] could not email the reset code: {e}")
            raise HTTPException(503, "Couldn't send the reset email right now. Please try again later.")
    return {"ok": True}


@router.post("/auth/reset")
def reset_password(body: Reset, conn=Depends(get_conn)):
    user = one_as_dict(conn.execute("SELECT * FROM users WHERE email = ?", (body.email.strip().lower(),)))
    if not user or not check_reset(conn, user["user_id"], body.code):
        raise HTTPException(400, "That code is wrong or has expired. Ask for a new one.")
    conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                 (hash_password(body.new_password), user["user_id"]))
    conn.commit()
    end_other_sessions(conn, user["user_id"])                  # whoever had the old password is signed out
    return {"token": start_session(conn, user["user_id"]), "user": public_user(user)}


@router.post("/auth/change-password")
def change_password(body: ChangePassword, user=Depends(current_user), token=Depends(current_token),
                    conn=Depends(get_conn)):
    if not check_password(body.current_password, user["password_hash"]):
        raise HTTPException(400, "Your current password is wrong.")
    if body.new_password == body.current_password:
        raise HTTPException(400, "The new password must be different from the current one.")
    conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                 (hash_password(body.new_password), user["user_id"]))
    conn.commit()
    end_other_sessions(conn, user["user_id"], keep_token=token)   # stay signed in here only
    return {"ok": True}


@router.post("/auth/logout")
def logout(token=Depends(current_token), conn=Depends(get_conn)):
    end_session(conn, token)
    return {"ok": True}


def _supervisor_code(conn, user, renew=False):
    """A supervisor's personal code; accounts made before codes existed get one the first time it's asked for."""
    code = None if renew else user.get("supervisor_code")
    if not code:
        code = codes.new_code(conn)
        conn.execute("UPDATE users SET supervisor_code = ? WHERE user_id = ?", (code, user["user_id"]))
        conn.commit()
    return code


@router.get("/auth/me")
def me(user=Depends(current_user), conn=Depends(get_conn)):
    out = public_user(user)
    if user["role"] == "supervisor":
        out["supervisor_code"] = _supervisor_code(conn, user)
    return out


@router.post("/auth/supervisor-code")
def regenerate_supervisor_code(user=Depends(current_user), conn=Depends(get_conn)):
    if user["role"] != "supervisor":
        raise HTTPException(403, "Only supervisors have a supervisor code.")
    return {"supervisor_code": _supervisor_code(conn, user, renew=True),
            "note": "Teams that already added you stay in your list."}


# ---------- Teams ----------

class NewTeam(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    repo_url: str | None = None


class JoinTeam(BaseModel):
    join_code: str


class AddSupervisor(BaseModel):
    supervisor_code: str


class Supervision(BaseModel):
    status: str | None = None
    archived: bool | None = None


class PanelReview(BaseModel):
    join_code: str


class EditTeam(BaseModel):
    name: str | None = Field(None, min_length=2, max_length=80)
    repo_url: str | None = None


class MyIdentities(BaseModel):
    git_identities: list[str] = Field(default_factory=list, max_length=10)


class LinkAuthor(BaseModel):
    user_id: str
    identity: str = Field(min_length=1, max_length=200)


def _clean_repo(url):
    url = (url or "").strip()
    if not url:
        return None
    if not sync.GITHUB_URL.match(url):
        raise HTTPException(400, "Repository link must look like https://github.com/owner/repo")
    return url.removesuffix("/").removesuffix(".git")


def _team_out(conn, t, my_role):
    out = {k: t[k] for k in ("team_id", "name", "repo_url", "sync_status", "sync_message", "last_synced")}
    out["is_research"] = bool(t["is_research"])
    out["label"] = RESEARCH_LABEL if t["is_research"] else None
    out["my_role"] = my_role
    out["anonymised"] = is_anonymised(t, my_role)
    if out["anonymised"]:
        out.update(name=anonymous_team_name(t["team_id"]), repo_url=None, sync_message=None)
    if my_role in ("lead", "member", "supervisor"):
        out["join_code"] = t["join_code"]
    if my_role == "supervisor":
        out["supervision_status"] = t.get("supervision_status")
        out["archived"] = bool(t.get("archived_at"))
    return out


def _live_team_by_code(conn, join_code):
    return one_as_dict(conn.execute("SELECT * FROM teams WHERE REPLACE(join_code, '-', '') = ? AND is_research = 0",
                                    (codes.normalise(join_code),)))


@router.get("/teams")
def list_teams(user=Depends(current_user), conn=Depends(get_conn)):
    teams = visible_teams(conn, user)
    counts = dict(conn.execute("SELECT team_id, COUNT(*) FROM activities GROUP BY team_id").fetchall())
    members = dict(conn.execute("SELECT team_id, COUNT(*) FROM team_members WHERE team_role != 'supervisor' "
                                "GROUP BY team_id").fetchall())
    out = []
    for t in teams:
        row = _team_out(conn, t, t["my_role"])
        row["activities"] = counts.get(t["team_id"], 0)
        row["members"] = members.get(t["team_id"], 0)
        d = service.drift(conn, t["team_id"])
        row["drift_severity"] = d["overall_severity"]
        out.append(row)
    return out


@router.post("/teams")
def create_team(body: NewTeam, user=Depends(current_user), conn=Depends(get_conn)):
    if user["role"] != "student":
        raise HTTPException(403, "Teams are created by students. Supervisors are added by a team with their supervisor code.")
    team_id = f"live__{secrets.token_hex(4)}"
    conn.execute("INSERT INTO teams (team_id, name, repo_url, is_research, join_code, created_by, created_at, "
                 "sync_status) VALUES (?, ?, ?, 0, ?, ?, ?, 'idle')",
                 (team_id, body.name.strip(), _clean_repo(body.repo_url), codes.new_code(conn), user["user_id"], now()))
    _add_member(conn, team_id, user)
    return _team_out(conn, one_as_dict(conn.execute("SELECT * FROM teams WHERE team_id = ?", (team_id,))), "lead")


def _add_member(conn, team_id, user, role=None):
    if user["role"] == "supervisor":
        role, code = "supervisor", None              # supervisors watch; their activity is not analysed
    else:
        role, code = role or "lead", sync.member_code(team_id, user["user_id"])
    conn.execute("INSERT INTO team_members (team_id, user_id, team_role, member_code, git_identities, joined_at) "
                 "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (team_id, user_id) DO NOTHING",
                 (team_id, user["user_id"], role, code, "[]", now()))
    conn.commit()


@router.post("/teams/join")
def join_team(body: JoinTeam, user=Depends(current_user), conn=Depends(get_conn)):
    if user["role"] != "student":
        raise HTTPException(403, "Supervisors don't join with a team code: the team adds you with your supervisor "
                                 "code. To look at another team once, use Panel review.")
    t = _live_team_by_code(conn, body.join_code)
    if not t:
        raise HTTPException(404, "No team has this join code.")
    _add_member(conn, t["team_id"], user, role="member")
    _, my_role = team_access(conn, user, t["team_id"])
    return _team_out(conn, t, my_role)


@router.get("/teams/{team_id}")
def team_detail(team_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    out = _team_out(conn, t, my_role)
    if t["is_research"] or out["anonymised"]:            # member codes only, no names / emails / git ids
        codes = [c for (c,) in conn.execute("SELECT DISTINCT member_id FROM activities WHERE team_id = ? "
                                            "AND member_id IS NOT NULL ORDER BY member_id", (team_id,)).fetchall()]
        out["members"] = [{"member_code": c, "name": None, "team_role": "member", "git_identities": []}
                          for c in codes]
        out["unlinked_authors"] = []
        return out
    rows = rows_as_dicts(conn.execute(
        "SELECT m.user_id, m.team_role, m.member_code, m.git_identities, u.name, u.email FROM team_members m "
        "JOIN users u ON u.user_id = m.user_id WHERE m.team_id = ? ORDER BY m.team_role, u.name", (team_id,)))
    out["members"] = [{"user_id": r["user_id"], "name": r["name"], "email": r["email"], "team_role": r["team_role"],
                       "member_code": r["member_code"], "git_identities": json.loads(r["git_identities"] or "[]"),
                       "is_me": r["user_id"] == user["user_id"]} for r in rows]
    out["unlinked_authors"] = json.loads(t.get("unlinked_authors") or "[]")
    return out


@router.put("/teams/{team_id}")
def edit_team(team_id: str, body: EditTeam, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    require_live(t, "edit it")
    require(my_role, ("lead", "supervisor"), "edit the team")
    if body.name is not None:
        conn.execute("UPDATE teams SET name = ? WHERE team_id = ?", (body.name.strip(), team_id))
    if body.repo_url is not None:
        conn.execute("UPDATE teams SET repo_url = ? WHERE team_id = ?", (_clean_repo(body.repo_url), team_id))
    conn.commit()
    return team_detail(team_id, user, conn)


@router.post("/teams/{team_id}/supervisors")
def add_supervisor(team_id: str, body: AddSupervisor, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    require_live(t, "add a supervisor")
    require(my_role, ("lead", "member"), "add a supervisor")
    supervisor = one_as_dict(conn.execute(
        "SELECT * FROM users WHERE REPLACE(supervisor_code, '-', '') = ? AND role = 'supervisor'",
        (codes.normalise(body.supervisor_code),)))
    if not supervisor:
        raise HTTPException(404, "No supervisor has this code. Check it with your supervisor.")
    if membership(conn, team_id, supervisor["user_id"]):
        conn.execute("UPDATE team_members SET archived_at = NULL WHERE team_id = ? AND user_id = ?",
                     (team_id, supervisor["user_id"]))
        conn.commit()
    else:
        _add_member(conn, team_id, supervisor)
    return team_detail(team_id, user, conn)


@router.delete("/teams/{team_id}/members/{member_user_id}")
def remove_member(team_id: str, member_user_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    require_live(t, "remove members")
    require(my_role, ("lead",), "remove members")
    target = membership(conn, team_id, member_user_id)
    if not target or target["team_role"] != "member":
        raise HTTPException(400, "Only team members can be removed (not the lead or a supervisor).")
    conn.execute("DELETE FROM team_members WHERE team_id = ? AND user_id = ?", (team_id, member_user_id))
    conn.commit()
    return team_detail(team_id, user, conn)


@router.put("/teams/{team_id}/supervision")
def set_supervision(team_id: str, body: Supervision, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    require(my_role, ("supervisor",), "change the supervision status")
    if "status" in body.model_fields_set:
        if body.status is not None and body.status not in SUPERVISION_STATUSES:
            raise HTTPException(400, f"Status must be one of {', '.join(SUPERVISION_STATUSES)}, or empty.")
        conn.execute("UPDATE team_members SET supervision_status = ? WHERE team_id = ? AND user_id = ?",
                     (body.status, team_id, user["user_id"]))
    if body.archived is not None:
        conn.execute("UPDATE team_members SET archived_at = ? WHERE team_id = ? AND user_id = ?",
                     (now() if body.archived else None, team_id, user["user_id"]))
    conn.commit()
    row = one_as_dict(conn.execute("SELECT supervision_status, archived_at FROM team_members "
                                   "WHERE team_id = ? AND user_id = ?", (team_id, user["user_id"])))
    return {"team_id": team_id, "supervision_status": row["supervision_status"], "archived": bool(row["archived_at"])}


# ---------- Panel review (a supervisor looks at another team, one at a time) ----------

def _panel_out(conn, user):
    row = one_as_dict(conn.execute("SELECT * FROM panel_reviews WHERE user_id = ? AND expires_at > ?",
                                   (user["user_id"], now())))
    if not row:
        return None
    t = one_as_dict(conn.execute("SELECT * FROM teams WHERE team_id = ?", (row["team_id"],)))
    out = _team_out(conn, t, PANEL_ROLE)
    out["expires_at"] = row["expires_at"]
    return out


def _require_supervisor(user):
    if user["role"] != "supervisor":
        raise HTTPException(403, "Panel review is for supervisors.")


@router.get("/panel-review")
def get_panel_review(user=Depends(current_user), conn=Depends(get_conn)):
    _require_supervisor(user)
    return _panel_out(conn, user)


@router.post("/panel-review")
def start_panel_review(body: PanelReview, user=Depends(current_user), conn=Depends(get_conn)):
    _require_supervisor(user)
    t = _live_team_by_code(conn, body.join_code)
    if not t:
        raise HTTPException(404, "No team has this join code.")
    if membership(conn, t["team_id"], user["user_id"]):
        raise HTTPException(400, "You already supervise this team - open it from your list.")
    conn.execute("INSERT INTO panel_reviews (user_id, team_id, started_at, expires_at) VALUES (?, ?, ?, ?) "
                 "ON CONFLICT (user_id) DO UPDATE SET team_id = excluded.team_id, started_at = excluded.started_at, "
                 "expires_at = excluded.expires_at",
                 (user["user_id"], t["team_id"], now(), now(timedelta(hours=PANEL_REVIEW_HOURS))))
    conn.commit()
    return _panel_out(conn, user)


@router.delete("/panel-review")
def end_panel_review(user=Depends(current_user), conn=Depends(get_conn)):
    _require_supervisor(user)
    conn.execute("DELETE FROM panel_reviews WHERE user_id = ?", (user["user_id"],))
    conn.commit()
    return {"ok": True}


@router.put("/teams/{team_id}/me")
def set_my_identities(team_id: str, body: MyIdentities, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    require_live(t, "link identities")
    require(my_role, ("lead", "member"), "link git identities")
    ids = sorted({i.strip() for i in body.git_identities if i.strip()})
    conn.execute("UPDATE team_members SET git_identities = ? WHERE team_id = ? AND user_id = ?",
                 (json.dumps(ids), team_id, user["user_id"]))
    conn.commit()
    return {"git_identities": ids, "note": "Run a sync so your commits are credited to you."}


@router.post("/teams/{team_id}/link")
def link_author(team_id: str, body: LinkAuthor, user=Depends(current_user), conn=Depends(get_conn)):
    """Lead / supervisor: credit an unlinked git author (name or email) to a member of the team,
    for when a student commits under an identity they never added themselves."""
    t, my_role = team_access(conn, user, team_id)
    require_live(t, "link identities")
    require(my_role, ("lead", "supervisor"), "link a git author to a member")
    m = one_as_dict(conn.execute("SELECT git_identities FROM team_members WHERE team_id = ? AND user_id = ? "
                                 "AND member_code IS NOT NULL", (team_id, body.user_id)))
    if not m:
        raise HTTPException(404, "That person is not a member of this team.")
    ids = sorted(set(json.loads(m["git_identities"] or "[]")) | {body.identity.strip()})
    conn.execute("UPDATE team_members SET git_identities = ? WHERE team_id = ? AND user_id = ?",
                 (json.dumps(ids), team_id, body.user_id))
    conn.commit()
    return {"git_identities": ids, "note": "Run a sync so these commits are credited to the member."}


# ---------- Live sync (Component 2 pipeline for a real team) ----------

def _sync_status(conn, team_id):
    t = one_as_dict(conn.execute("SELECT sync_status, sync_message, sync_started, last_synced FROM teams "
                                 "WHERE team_id = ?", (team_id,)))
    return {"status": t["sync_status"] or "idle", "message": t["sync_message"],
            "started": t["sync_started"], "last_synced": t["last_synced"]}


@router.post("/teams/{team_id}/sync")
def start_sync(team_id: str, background: BackgroundTasks, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    require_live(t, "sync it")
    require(my_role, ("lead", "member", "supervisor"), "sync the team")
    status = _sync_status(conn, team_id)
    if status["status"] == "running" and status["started"]:
        started = datetime.strptime(status["started"], TS_FORMAT)
        if datetime.utcnow() - started < timedelta(minutes=sync.STALE_AFTER_MINUTES):
            raise HTTPException(409, "A sync is already running for this team.")
    conn.execute("UPDATE teams SET sync_status = 'running', sync_message = 'Starting...', sync_started = ? "
                 "WHERE team_id = ?", (now(), team_id))
    conn.commit()
    background.add_task(sync.run_sync, team_id)
    return _sync_status(conn, team_id)


@router.get("/teams/{team_id}/sync")
def sync_status(team_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    status = _sync_status(conn, team_id)
    if is_anonymised(t, my_role):
        status["message"] = None
    return status


# ---------- Blueprint (stand-in until Component 1 provides it) ----------

class Blueprint(BaseModel):
    target: str | None = Field(None, max_length=120)
    algorithm: str | None = Field(None, max_length=120)
    metric: str | None = Field(None, max_length=120)
    notes: str | None = Field(None, max_length=2000)


@router.get("/teams/{team_id}/blueprint")
def get_blueprint(team_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    if t["is_research"]:
        path = BLUEPRINT_DIR / f"{team_id}.json"
        bp = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        source = "inferred" if bp.get("_inferred_from_early_activity") else ("file" if bp else "none")
        return {"source": source, "editable": False, "notes": None, "updated_at": None, "updated_by": None,
                **{d: (bp.get(d) or {}).get("name") for d in ("target", "algorithm", "metric")}}
    bp = one_as_dict(conn.execute("SELECT b.*, u.name AS updated_by_name FROM blueprints b "
                                  "LEFT JOIN users u ON u.user_id = b.updated_by WHERE b.team_id = ?", (team_id,)))
    base = {"source": "app" if bp else "none", "editable": my_role in ("lead", "member")}
    if not bp:
        return {**base, "target": None, "algorithm": None, "metric": None, "notes": None,
                "updated_at": None, "updated_by": None}
    out = {**base, **{k: bp[k] for k in ("target", "algorithm", "metric", "notes", "updated_at")},
           "updated_by": bp["updated_by_name"]}
    if is_anonymised(t, my_role):                       # free-text notes and the editor's name can identify people
        out.update(notes=None, updated_by=None)
    return out


@router.put("/teams/{team_id}/blueprint")
def put_blueprint(team_id: str, body: Blueprint, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    require_live(t, "change the blueprint")
    require(my_role, ("lead", "member"), "change the blueprint")
    vals = [(v or "").strip() or None for v in (body.target, body.algorithm, body.metric, body.notes)]
    conn.execute("INSERT INTO blueprints VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (team_id) DO UPDATE SET "
                 "target = excluded.target, algorithm = excluded.algorithm, metric = excluded.metric, "
                 "notes = excluded.notes, updated_by = excluded.updated_by, updated_at = excluded.updated_at",
                 (team_id, *vals, user["user_id"], now()))
    conn.commit()
    if conn.execute("SELECT 1 FROM activities WHERE team_id = ? LIMIT 1", (team_id,)).fetchone():
        sync.analyse_team(conn, team_id)              # drift is re-checked against the new plan straight away
    return get_blueprint(team_id, user, conn)


# ---------- Chat ----------

class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    reply_to: str | None = None


@router.get("/teams/{team_id}/chat")
def get_chat(team_id: str, limit: int = Query(200, ge=1, le=1000),
             user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    if t["is_research"]:
        rows = conn.execute("SELECT activity_id, member_id, ts, text, meta FROM activities WHERE team_id = ? "
                            "AND source = 'chat' ORDER BY ts DESC LIMIT ?", (team_id, limit)).fetchall()
        msgs = [{"msg_id": aid.rsplit(":", 1)[-1], "sender": code, "member_code": code, "ts": ts, "text": text,
                 "reply_to": (json.loads(meta or "{}").get("reply_to") or "").rsplit(":", 1)[-1] or None,
                 "mine": False, "team_role": "member"} for aid, code, ts, text, meta in rows]
        return {"read_only": True, "simulated": True, "messages": msgs[::-1]}
    # A real team's chat is private to the team: not its supervisor, not examiners
    require(my_role, ("lead", "member"), "read the team chat (its stage mix is in the Behaviour tab)")
    rows = rows_as_dicts(conn.execute(
        "SELECT c.msg_id, c.ts, c.text, c.reply_to, c.user_id, u.name, m.member_code, m.team_role "
        "FROM chat_messages c JOIN users u ON u.user_id = c.user_id "
        "LEFT JOIN team_members m ON m.team_id = c.team_id AND m.user_id = c.user_id "
        "WHERE c.team_id = ? ORDER BY c.ts DESC LIMIT ?", (team_id, limit)))
    msgs = [{"msg_id": r["msg_id"], "sender": r["name"], "member_code": r["member_code"], "ts": r["ts"],
             "text": r["text"], "reply_to": r["reply_to"], "mine": r["user_id"] == user["user_id"],
             "team_role": r["team_role"]} for r in rows]
    msgs = merge_chat(conn, team_id, msgs[::-1], limit)       # oldest first, with Buddy's messages added
    return {"read_only": False, "simulated": False, "messages": msgs}


@router.post("/teams/{team_id}/chat")
def post_chat(team_id: str, body: ChatIn, background: BackgroundTasks, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    require_live(t, "post in its chat")
    require(my_role, ("lead", "member"), "post in the chat")
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Message is empty.")
    reply_to = body.reply_to if body.reply_to and conn.execute(
        "SELECT 1 FROM chat_messages WHERE msg_id = ? AND team_id = ?", (body.reply_to, team_id)).fetchone() else None
    msg_id, ts = new_id("m"), now()
    conn.execute("INSERT INTO chat_messages VALUES (?, ?, ?, ?, ?, ?)", (msg_id, team_id, user["user_id"], ts, text, reply_to))
    conn.commit()

    # Members' messages are also Component 2 activities (like the research chat); supervisors' are not analysed
    m = one_as_dict(conn.execute("SELECT member_code FROM team_members WHERE team_id = ? AND user_id = ?",
                                 (team_id, user["user_id"])))
    if m and m["member_code"]:
        aid = f"chat:{team_id}:{msg_id}"
        save_activities(conn, [{"activity_id": aid, "team_id": team_id, "source": "chat",
                                "member_id": m["member_code"], "ts": ts, "text": text,
                                "meta": {"reply_to": f"chat:{team_id}:{reply_to}" if reply_to else None,
                                         "in_app": True}}])
        stage, conf = classify(text)
        save_labels(conn, "baseline", [(aid, stage, conf)])      # trained model re-labels it at the next sync
        run_in_background(background, team_id, msg_id, m["member_code"], user["user_id"], ts, text,
                          reply_to=reply_to)   # Component 1 (Buddy); reply_to helps it link "agreed" to a proposal
    return {"msg_id": msg_id, "ts": ts}


# ---------- Behaviour (Component 2 output, behind the access check) ----------

def _roster(conn, team):
    """Member codes of everyone who signed up to a real team (supervisors are not analysed)."""
    if team["is_research"]:
        return []
    return [c for (c,) in conn.execute("SELECT member_code FROM team_members WHERE team_id = ? "
                                       "AND member_code IS NOT NULL", (team["team_id"],)).fetchall()]


def _has_data(conn, team_id):
    if not conn.execute("SELECT 1 FROM activities WHERE team_id = ? LIMIT 1", (team_id,)).fetchone():
        raise HTTPException(404, "No activity yet. Add the repository link and run a sync, or start chatting.")


@router.get("/teams/{team_id}/behavior/summary")
def behavior_summary(team_id: str, days: int = Query(30, ge=1, le=365),
                     user=Depends(current_user), conn=Depends(get_conn)):
    t, _ = team_access(conn, user, team_id)
    _has_data(conn, team_id)
    s = service.team_summary(conn, team_id, days, roster=_roster(conn, t))
    if s is None:
        raise HTTPException(404, "Only chat so far - sync the repository to see the work breakdown.")
    return s


@router.get("/teams/{team_id}/behavior/roles")
def behavior_roles(team_id: str, window: str = Query("7d", pattern="^(3d|7d|30d|all)$"),
                   user=Depends(current_user), conn=Depends(get_conn)):
    t, _ = team_access(conn, user, team_id)
    _has_data(conn, team_id)
    return service.member_roles(conn, team_id, window, roster=_roster(conn, t))


# ---------- Feedback (for the supervisor / team evaluation) ----------

class FeedbackIn(BaseModel):
    team_id: str | None = None
    page: str | None = Field(None, max_length=100)
    rating: int | None = Field(None, ge=1, le=5)
    comment: str | None = Field(None, max_length=4000)


@router.post("/feedback")
def post_feedback(body: FeedbackIn, user=Depends(current_user), conn=Depends(get_conn)):
    if not body.rating and not (body.comment or "").strip():
        raise HTTPException(400, "Give a rating or write a comment.")
    team_id = body.team_id
    if team_id:
        team_access(conn, user, team_id)
    conn.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (new_id("f"), user["user_id"], team_id, body.page, body.rating, (body.comment or "").strip(), now()))
    conn.commit()
    return {"ok": True}


def _feedback_rows(conn, user):
    """Examiners see all feedback; supervisors see feedback from their teams plus their own."""
    sql = ("SELECT f.ts, u.name, u.role, f.team_id, t.name AS team_name, f.page, f.rating, f.comment "
           "FROM feedback f JOIN users u ON u.user_id = f.user_id LEFT JOIN teams t ON t.team_id = f.team_id ")
    if user["role"] == "examiner":
        rows = rows_as_dicts(conn.execute(sql + "ORDER BY f.ts DESC"))
        research = {t for (t,) in conn.execute("SELECT team_id FROM teams WHERE is_research = 1").fetchall()}
        for r in rows:                     # examiners see no names: the writer by role, real teams by code name
            r["name"] = f"A {r['role']}"
            if r["team_id"] and r["team_id"] not in research:
                r["team_name"] = anonymous_team_name(r["team_id"])
        return rows
    if user["role"] == "supervisor":
        rows = rows_as_dicts(conn.execute(
            sql + "WHERE f.user_id = ? OR f.team_id IN (SELECT team_id FROM team_members WHERE user_id = ? "
                  "AND team_role = 'supervisor') ORDER BY f.ts DESC", (user["user_id"], user["user_id"])))
        for r in rows:                     # students' feedback stays anonymous to their supervisor
            if r["role"] == "student":
                r["name"] = "A student"
        return rows
    raise HTTPException(403, "Only supervisors and examiners can read feedback.")


@router.get("/feedback")
def list_feedback(user=Depends(current_user), conn=Depends(get_conn)):
    rows = _feedback_rows(conn, user)

    def stats(items):
        rated = [r["rating"] for r in items if r["rating"]]
        return {"count": len(items), "average_rating": round(sum(rated) / len(rated), 2) if rated else None}

    roles = sorted({r["role"] for r in rows})
    return {**stats(rows), "by_role": {k: stats([r for r in rows if r["role"] == k]) for k in roles}, "items": rows}


@router.get("/feedback.csv")
def feedback_csv(user=Depends(current_user), conn=Depends(get_conn)):
    rows = _feedback_rows(conn, user)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["ts", "name", "role", "team_id", "team_name", "page", "rating", "comment"])
    w.writeheader()
    w.writerows(rows)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=buddy_feedback.csv"})
