"""Document upload and document suggestions, for the Blueprint tab.

    POST   /app/teams/{team_id}/documents                         upload (lead, member); read in the background
    GET    /app/teams/{team_id}/documents                         list the team's documents and their status
    DELETE /app/teams/{team_id}/documents/{doc_id}                remove a document and its open suggestions
    POST   /app/teams/{team_id}/plan/candidates/{fact_id}/accept  put a suggestion into the blueprint
    POST   /app/teams/{team_id}/plan/candidates/{fact_id}/dismiss never suggest that value again

Only the extracted text is kept, never the file itself. Uploading needs the python-multipart package.
"""
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from buddy.access import is_anonymised, membership, require, require_live, team_access
from buddy.auth import current_user, get_conn

from ..documents.parse import MAX_BYTES, DocumentError, kind_of
from ..plan import build_plan
from ..service import speaker_for
from ..storage import create_document, delete_document, ensure_c1_tables, list_documents

log = logging.getLogger("component1")
router = APIRouter(tags=["Component 1 (Buddy): documents"])

EDITORS = ("lead", "member")
_tables_ready = False


def _ready(conn):
    global _tables_ready
    if not _tables_ready:
        ensure_c1_tables(conn)
        _tables_ready = True


def _public(doc: dict) -> dict:
    return {k: doc.get(k) for k in ("doc_id", "filename", "size_bytes", "status", "error", "pages", "facts",
                                    "uploaded_at", "processed_at")}


def _member_code(conn, team_id: str, user: dict) -> str:
    m = membership(conn, team_id, user["user_id"])
    return speaker_for(m.get("member_code") if m else None, user["user_id"])


def _service():
    from ..hook import service      # the one shared service, created on first use
    return service()


def _process(team_id: str, doc_id: str, filename: str, data: bytes) -> None:
    _service().process_document(team_id, doc_id, filename, data)


@router.post("/teams/{team_id}/documents")
def upload_document(team_id: str, background: BackgroundTasks, file: UploadFile = File(...),
                    user=Depends(current_user), conn=Depends(get_conn)):
    team, my_role = team_access(conn, user, team_id)
    require_live(team, "upload documents")
    require(my_role, EDITORS, "upload documents")
    filename = (file.filename or "").strip() or "document"
    try:
        kind_of(filename)
    except DocumentError as exc:
        raise HTTPException(415, str(exc))
    data = file.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    _ready(conn)
    doc = create_document(conn, team_id, filename, len(data), uploaded_by=user["user_id"])
    background.add_task(_process, team_id, doc["doc_id"], filename, data)
    return {"document": _public(doc)}


@router.get("/teams/{team_id}/documents")
def get_documents(team_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    team, my_role = team_access(conn, user, team_id)
    if team["is_research"]:
        return {"available": False, "documents": []}
    _ready(conn)
    docs = [_public(d) for d in list_documents(conn, team_id)]
    if is_anonymised(team, my_role):                     # file names can identify people
        docs = [dict(d, filename=f"Document {i + 1}") for i, d in enumerate(docs)]
    return {"available": True, "can_upload": my_role in EDITORS and not team["is_research"], "documents": docs}


@router.delete("/teams/{team_id}/documents/{doc_id}")
def remove_document(team_id: str, doc_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    team, my_role = team_access(conn, user, team_id)
    require_live(team, "remove documents")
    require(my_role, EDITORS, "remove documents")
    _ready(conn)
    if not delete_document(conn, team_id, doc_id):
        raise HTTPException(404, "No such document.")
    _service().refresh_suggestions(team_id)
    return {"ok": True}


def _decide(team_id: str, fact_id: str, action: str, user: dict, conn) -> dict:
    team, my_role = team_access(conn, user, team_id)
    require_live(team, f"{action} suggestions")
    require(my_role, EDITORS, f"{action} suggestions")
    _ready(conn)
    who = _member_code(conn, team_id, user)
    svc = _service()
    result = svc.accept_suggestion(team_id, fact_id, who) if action == "accept" else \
        svc.dismiss_suggestion(team_id, fact_id, who)
    if not result.get("ok"):
        raise HTTPException(409, result.get("error") or "The suggestion could not be changed.")
    return {**result, "plan": build_plan(conn, team_id, include_evidence=True)}


@router.post("/teams/{team_id}/plan/candidates/{fact_id}/accept")
def accept_candidate(team_id: str, fact_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    return _decide(team_id, fact_id, "accept", user, conn)


@router.post("/teams/{team_id}/plan/candidates/{fact_id}/dismiss")
def dismiss_candidate(team_id: str, fact_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    return _decide(team_id, fact_id, "dismiss", user, conn)
