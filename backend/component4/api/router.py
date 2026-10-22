"""Component 4's web API. All addresses start with /c4.

  GET  /c4/health     is it running, and is a trained model available?
  GET  /c4/features   the features the model expects, with labels and scales
  POST /c4/predict    one project's features -> probability, explanation, forecast, recommendations
  GET  /c4/demo       the same for a held-out synthetic project (for the Predict screen before real data)
  GET  /c4/teams/{team_id}/prediction   a real team's prediction from its saved Component 2 and 3 data and its
                                       repo's readiness; no number until enough signals come from the team
  GET|POST /c4/teams/{team_id}/sheets   saved score sheets per member (anyone in the group); DELETE .../sheets/{uid}
  POST /c4/teams/{team_id}/helpers      saved sheets -> who could help with each weak stage
  GET|POST /c4/teams/{team_id}/milestone-results   the group's own result after each milestone; DELETE .../{id}
  GET  /c4/supervisor/outlook           suggested status dot for each group in the supervisor's list

Academic marks never reach the prediction: they are only used by the helper suggestions and are never shown.
"""
import base64
import binascii
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from buddy.access import require, require_live, team_access
from buddy.auth import current_user, get_conn

from ..common import config
from ..m1_feature_integration.team_sources import team_project
from ..m6_improvement import history
from ..m6_improvement import milestone_results
from ..m6_improvement.competency.scoring import competency, suggest_helpers
from ..m6_improvement.competency import store as sheet_store
from ..m6_improvement.competency.sheets import SheetError, parse_sheet
from ..m6_improvement.weak_stages import weak_stages
from ..synthetic.feature_spec import FEATURES
from . import outlook, service

router = APIRouter(prefix=config.API["prefix"], tags=["Component 4 - Success prediction"])


class PredictRequest(BaseModel):
    features: dict[str, float | None] = Field(default_factory=dict)
    recurring_causes: list[str] = Field(default_factory=list)
    include_forecast: bool = True


def _ready():
    try:
        return service.load_bundle()
    except service.ModelNotReady as error:
        raise HTTPException(503, str(error))


@router.get("/health")
def health():
    try:
        bundle = service.load_bundle()
        return {"component": "C4", "status": "ok", "model_ready": True, "model_version": bundle.get("version")}
    except service.ModelNotReady as error:
        return {"component": "C4", "status": "ok", "model_ready": False, "detail": str(error)}


@router.get("/features")
def features():
    return [{"name": s.name, "label": config.FEATURE_LABELS.get(s.name, s.name), "source": s.source,
             "kind": s.kind, "scale": config.KIND_TEXT[s.kind], "status": s.status} for s in FEATURES]


@router.post("/predict")
def predict(body: PredictRequest):
    known = {s.name for s in FEATURES}
    unknown = sorted(set(body.features) - known)
    if unknown:
        raise HTTPException(422, f"Unknown features: {', '.join(unknown)}. See /c4/features.")
    bad_causes = sorted(set(body.recurring_causes) - set(config.FAILURE_PATTERNS["causes"]))
    if bad_causes:
        raise HTTPException(422, f"Unknown root causes: {', '.join(bad_causes)}.")
    bundle = _ready()
    return service.predict(body.features, body.recurring_causes, "live", body.include_forecast, bundle)


@router.get("/demo")
def demo():
    bundle = _ready()
    return service.predict(service.demo_project(bundle), source="synthetic", bundle=bundle)


@router.get("/teams/{team_id}/prediction")
def team_prediction(team_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    team_access(conn, user, team_id)
    bundle = _ready()
    project = team_project(conn, team_id)
    readiness = project["readiness"]
    signals = len(FEATURES) - len(project["missing"])
    about = {"team_id": team_id, "signals_from_team": signals, "total_signals": len(FEATURES),
             "min_signals": config.API["min_team_signals"], "lagging_stages": project["lagging_stages"],
             "milestones": project["milestones"], "timeline_basis": project["timeline_basis"]}

    if signals < config.API["min_team_signals"]:
        return {**about, "enough_data": False, "source": "live", "model_version": bundle.get("version"),
                "evaluated_at": service.now(), "probability": None, "factors": [], "forecast": [],
                "recommendations": [], "readiness": readiness, "history": [], "changes": None,
                "imputed_features": project["missing"],
                "weak_stages": weak_stages(project["lagging_stages"], project["recurring_causes"], [], readiness)}

    result = service.predict(project["features"], project["recurring_causes"], "live", True, bundle, readiness,
                             project["recurring_counts"])
    history.record(conn, team_id, result, project["features"])
    saved = history.recent(conn, team_id, result["model_version"])
    return {**result, **about, "enough_data": True,
            "history": [row["probability"] for row in saved], "changes": history.changes(saved),
            "weak_stages": weak_stages(project["lagging_stages"], project["recurring_causes"],
                                       result["recommendations"], readiness)}


class Sheet(BaseModel):
    member_user_id: str
    content_base64: str


class HelpersRequest(BaseModel):
    stages: list[str] | None = None


def _decode(sheet: Sheet) -> bytes:
    try:
        data = base64.b64decode(sheet.content_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(422, "A score sheet could not be read. Upload the PDF again.")
    if len(data) > config.COMPETENCY["max_sheet_bytes"]:
        raise HTTPException(413, "A score sheet is too large.")
    return data


def _group(conn, user, team_id, action):
    team, my_role = team_access(conn, user, team_id)
    require_live(team, action)
    require(my_role, tuple(config.COMPETENCY["group_roles"]), action)
    return team


def _members(conn, team_id) -> dict[str, str]:
    roles = config.COMPETENCY["member_roles"]
    marks = ", ".join("?" for _ in roles)
    return dict(conn.execute(f"SELECT m.user_id, u.name FROM team_members m JOIN users u ON u.user_id = m.user_id "
                             f"WHERE m.team_id = ? AND m.team_role IN ({marks}) ORDER BY u.name", (team_id, *roles)).fetchall())


def _sheet_list(conn, team_id) -> list[dict]:
    dates = sheet_store.saved_dates(conn, team_id)
    return [{"user_id": uid, "name": name, "saved": uid in dates, "uploaded_at": dates.get(uid)}
            for uid, name in _members(conn, team_id).items()]


@router.get("/teams/{team_id}/sheets")
def list_sheets(team_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    _group(conn, user, team_id, "see saved score sheets")
    return _sheet_list(conn, team_id)


@router.post("/teams/{team_id}/sheets")
def upload_sheet(team_id: str, body: Sheet, user=Depends(current_user), conn=Depends(get_conn)):
    _group(conn, user, team_id, "upload score sheets")
    names = _members(conn, team_id)
    if body.member_user_id not in names:
        raise HTTPException(422, "A sheet must belong to a member of this group.")
    try:
        modules = parse_sheet(_decode(body))
    except SheetError as error:
        raise HTTPException(422, f"{names[body.member_user_id]}'s sheet: {error}")
    sheet_store.save(conn, team_id, body.member_user_id, modules, user["user_id"])
    return _sheet_list(conn, team_id)


@router.delete("/teams/{team_id}/sheets/{member_user_id}")
def delete_sheet(team_id: str, member_user_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    _group(conn, user, team_id, "remove score sheets")
    sheet_store.delete(conn, team_id, member_user_id)
    return _sheet_list(conn, team_id)


@router.post("/teams/{team_id}/helpers")
def team_helpers(team_id: str, body: HelpersRequest, user=Depends(current_user), conn=Depends(get_conn)):
    _group(conn, user, team_id, "suggest helpers")
    names = _members(conn, team_id)
    saved = {uid: modules for uid, modules in sheet_store.load_all(conn, team_id).items() if uid in names}
    if not saved:
        raise HTTPException(422, "No score sheets saved for this group yet. Upload at least one first.")

    if body.stages is None:
        project = team_project(conn, team_id)
        stages = [w["stage"] for w in weak_stages(project["lagging_stages"], project["recurring_causes"], [],
                                                  project["readiness"])]
    else:
        stages = body.stages
    unknown = sorted(set(stages) - set(config.STAGE_LABELS))
    if unknown:
        raise HTTPException(422, f"Unknown lifecycle stages: {', '.join(unknown)}.")

    members = {uid: {"name": names[uid], "competency": competency(modules)} for uid, modules in saved.items()}
    return {
        "team_id": team_id,
        "stages_source": "chosen" if body.stages is not None else "detected",
        "suggestions": suggest_helpers(stages, members, config.STAGE_LABELS),
        "sheets_used": len(members),
        "note": "Marks are never shown, only the suggested person.",
    }


class MilestoneResult(BaseModel):
    milestone: str = Field(min_length=1, max_length=config.MILESTONE_RESULTS["max_name"])
    milestone_date: date
    result: str


@router.get("/teams/{team_id}/milestone-results")
def list_milestone_results(team_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    _group(conn, user, team_id, "see milestone results")
    return milestone_results.listed(conn, team_id)


@router.post("/teams/{team_id}/milestone-results")
def add_milestone_result(team_id: str, body: MilestoneResult, user=Depends(current_user), conn=Depends(get_conn)):
    team, my_role = team_access(conn, user, team_id)
    require_live(team, "record milestone results")
    require(my_role, tuple(config.MILESTONE_RESULTS["record_roles"]), "record milestone results")
    if body.result not in config.MILESTONE_RESULTS["options"]:
        raise HTTPException(422, f"Result must be one of: {', '.join(config.MILESTONE_RESULTS['options'])}.")
    milestone_results.add(conn, team_id, body.milestone.strip(), body.milestone_date.isoformat(), body.result, user["user_id"])
    return milestone_results.listed(conn, team_id)


@router.delete("/teams/{team_id}/milestone-results/{result_id}")
def delete_milestone_result(team_id: str, result_id: str, user=Depends(current_user), conn=Depends(get_conn)):
    team, my_role = team_access(conn, user, team_id)
    require_live(team, "remove milestone results")
    require(my_role, tuple(config.MILESTONE_RESULTS["record_roles"]), "remove milestone results")
    milestone_results.remove(conn, team_id, result_id)
    return milestone_results.listed(conn, team_id)


@router.get("/supervisor/outlook")
def supervisor_outlook(user=Depends(current_user), conn=Depends(get_conn)):
    if user["role"] != "supervisor":
        raise HTTPException(403, "Only supervisors have a group list.")
    bundle = _ready()
    supervised = [team_id for (team_id,) in conn.execute(
        "SELECT team_id FROM team_members WHERE user_id = ? AND team_role = 'supervisor' AND archived_at IS NULL",
        (user["user_id"],)).fetchall()]
    research = [team_id for (team_id,) in conn.execute("SELECT team_id FROM teams WHERE is_research = 1").fetchall()]
    return [outlook.team_outlook(conn, team_id, bundle) for team_id in dict.fromkeys(supervised + research)]
