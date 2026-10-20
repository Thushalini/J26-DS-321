"""Component 3's web API. All addresses start with /c3 and, like the Buddy.ds routes, need a signed-in user.

  GET  /c3/health                                       is it running?
  GET  /c3/teams/{team_id}/alerts                       the team's alerts (anyone who can see the team)
  POST /c3/teams/{team_id}/alerts/{run_id}/resolve      mark resolved / reopen (team lead or member)
  POST /c3/teams/{team_id}/runs                         score a run and save its alert (collectors and scripts)

Same access rules as the rest of the app: team_access() decides who can see a team, require() who may
change things, and require_live() keeps the read-only research teams read-only.

Added to backend/main.py:
    from component3.api.router import router as c3_router
    app.include_router(c3_router)
"""
import os
import secrets

from fastapi import APIRouter, Body, Depends, Header, HTTPException
from pydantic import BaseModel

from buddy.access import require, require_live, team_access
from buddy.auth import current_user, get_conn

from . import service

router = APIRouter(prefix="/c3", tags=["Component 3 - Collaboration intelligence"])


class ResolveBody(BaseModel):
    resolved: bool = True


@router.get("/health")
def health():
    return {"component": "C3", "status": "ok"}


@router.get("/teams/{team_id}/alerts")
def get_alerts(team_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    team_access(conn, user, team_id)                      # 404 / 403 if this user may not see the team
    return service.alerts_payload(conn, team_id)


@router.post("/teams/{team_id}/alerts/{run_id}/resolve")
def resolve(team_id: str, run_id: str, body: ResolveBody, user=Depends(current_user), conn=Depends(get_conn)):
    t, my_role = team_access(conn, user, team_id)
    require_live(t, "resolve alerts")                     # not for the read-only research teams
    require(my_role, ("lead", "member"), "resolve alerts")
    alert = service.resolve_alert(conn, team_id, run_id, body.resolved, user["user_id"])
    if alert is None:
        raise HTTPException(status_code=404, detail=f"No alert for run '{run_id}' in this team")
    return alert


@router.post("/teams/{team_id}/runs")
def post_run(team_id: str, run: dict = Body(...), x_c3_key: str | None = Header(default=None),
             conn=Depends(get_conn)):
    """Scores a run. For collectors and scripts, not for the browser: it needs the key C3_API_KEY
    from backend/.env in the X-C3-Key header, and the team must exist."""
    expected = os.environ.get("C3_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Run scoring is switched off (C3_API_KEY is not set).")
    if not x_c3_key or not secrets.compare_digest(x_c3_key, expected):
        raise HTTPException(status_code=401, detail="Missing or wrong X-C3-Key header.")
    if not conn.execute("SELECT 1 FROM teams WHERE team_id = ?", (team_id,)).fetchone():
        raise HTTPException(status_code=404, detail=f"No team '{team_id}'")
    try:
        return service.ingest_run(conn, team_id, run)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
