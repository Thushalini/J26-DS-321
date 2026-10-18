"""
Step 5a: Component 2's web API (FR08). All addresses start with /c2.

  GET /c2/health                          -> is it running?
  GET /c2/teams                           -> which teams have data
  GET /c2/teams/{team_id}/summary         -> full output for C3 + C4 (agreed JSON)
  GET /c2/teams/{team_id}/drift           -> drift only
  GET /c2/teams/{team_id}/members         -> role vectors + role history
  GET /c2/teams/{team_id}/stages/daily    -> activity per stage per day (for C3)
  GET /c2/teams/{team_id}/roles?window=7d -> every member's stage mix in a window (3d, 7d, 30d, all)

Every request opens its own database connection and closes it at the end.
"""
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException, Query

from component2.api import service
from component2.db import connect

router = APIRouter(prefix="/c2", tags=["Component 2 - Behavioural analytics"])


@contextmanager
def db():
    conn = connect(create_tables=False)
    try:
        yield conn
    finally:
        conn.close()


def _known_team(conn, team_id):
    if not conn.execute("SELECT 1 FROM activities WHERE team_id = ? LIMIT 1", (team_id,)).fetchone():
        raise HTTPException(status_code=404, detail=f"No data for team '{team_id}'")


@router.get("/health")
def health():
    return {"component": "C2", "status": "ok"}


@router.get("/teams")
def teams():
    with db() as conn:
        return service.list_teams(conn)


@router.get("/teams/{team_id}/summary")
def summary(team_id: str, days: int = Query(30, ge=1, le=365, description="Size of the recent window in days")):
    with db() as conn:
        _known_team(conn, team_id)
        return service.team_summary(conn, team_id, days)


@router.get("/teams/{team_id}/drift")
def drift(team_id: str):
    with db() as conn:
        _known_team(conn, team_id)
        return service.drift(conn, team_id)


@router.get("/teams/{team_id}/members")
def members(team_id: str):
    with db() as conn:
        _known_team(conn, team_id)
        return service.members(conn, team_id)


@router.get("/teams/{team_id}/stages/daily")
def stages_daily(team_id: str, days: int = Query(30, ge=1, le=365)):
    with db() as conn:
        _known_team(conn, team_id)
        return service.daily_stage_counts(conn, team_id, days)

@router.get("/teams/{team_id}/roles")
def roles(team_id: str, window: str = Query("7d", pattern="^(3d|7d|30d|all)$",
                                            description="3d, 7d, 30d or all (whole project)")):
    with db() as conn:
        _known_team(conn, team_id)
        return service.member_roles(conn, team_id, window)