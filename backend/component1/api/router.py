"""The Blueprint tab's data, plus the document routes (documents.py).

    GET /app/teams/{team_id}/plan                      the blueprint now, with the 10 latest changes
    GET /app/teams/{team_id}/plan/history?field=...    every change, newest first (one field, or all)
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from buddy.access import is_anonymised, team_access
from buddy.auth import current_user, get_conn

from ..common.schema import FieldName
from ..history import field_history
from ..plan import build_plan
from ..storage import ensure_c1_tables, load_events

log = logging.getLogger("component1")
router = APIRouter(prefix="/app", tags=["Component 1 (Buddy): living blueprint"])

try:   # uploads need python-multipart; without it only the document routes are missing
    from .documents import router as _documents_router
    router.include_router(_documents_router)
except Exception as exc:
    log.warning("Component 1 document routes not loaded (pip install python-multipart): %s", exc)
    print(f"[component1] document routes not loaded: {exc}")

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
        anonymised = is_anonymised(team, my_role)              # file names can identify people
        plan = build_plan(conn, team_id, include_evidence=full, include_documents=not anonymised)
    except Exception:
        log.exception("Component 1 could not build the blueprint for team %s", team_id)
        _rollback(conn)
        raise HTTPException(503, "The blueprint is not available right now. Please try again.")
    return {"available": True, "can_see_sources": full, "can_edit": full, **plan}


@router.get("/teams/{team_id}/plan/history")
def get_plan_history(team_id: str, field: Optional[str] = Query(None), limit: int = Query(200, ge=1, le=1000),
                     user=Depends(current_user), conn=Depends(get_conn)):
    global _tables_ready
    team, my_role = team_access(conn, user, team_id)
    if team["is_research"]:
        return {"available": False, "entries": [], "total": 0}
    if field is not None and field not in {n.value for n in FieldName}:
        raise HTTPException(404, f"Unknown field '{field}'.")
    try:
        if not _tables_ready:
            ensure_c1_tables(conn)
            _tables_ready = True
        entries = field_history(load_events(conn, team_id), field=field,
                                include_evidence=my_role in ("lead", "member"),
                                hide_documents=is_anonymised(team, my_role))
    except Exception:
        log.exception("Component 1 could not build the history for team %s", team_id)
        _rollback(conn)
        raise HTTPException(503, "The history is not available right now. Please try again.")
    newest_first = list(reversed(entries))
    return {"available": True, "field": field, "total": len(entries), "entries": newest_first[:limit]}