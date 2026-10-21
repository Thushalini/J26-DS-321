"""The Blueprint tab's data: GET /app/teams/{team_id}/plan."""
import logging

from fastapi import APIRouter, Depends, HTTPException

from buddy.access import team_access
from buddy.auth import current_user, get_conn

from ..plan import build_plan
from ..storage import ensure_c1_tables

log = logging.getLogger("component1")
router = APIRouter(prefix="/app", tags=["Component 1 (Buddy): living blueprint"])

_tables_ready = False


def _rollback(conn):
    rollback = getattr(getattr(conn, "raw", None), "rollback", None)
    if rollback:
        try:
            rollback()
        except Exception:
            pass


@router.get("/teams/{team_id}/plan")
def get_plan(team_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    global _tables_ready
    team, my_role = team_access(conn, user, team_id)
    if team["is_research"]:
        return {"available": False, "reason": "Research teams have no live blueprint."}
    full = my_role in ("lead", "member")
    try:
        if not _tables_ready:
            ensure_c1_tables(conn)
            _tables_ready = True
        plan = build_plan(conn, team_id, include_evidence=full)
    except Exception:
        log.exception("Component 1 could not build the blueprint for team %s", team_id)
        _rollback(conn)
        raise HTTPException(503, "The blueprint is not available right now. Please try again.")
    return {"available": True, "can_see_sources": full, **plan}