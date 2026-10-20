"""Component 4's web API. All addresses start with /c4.

  GET  /c4/health     is it running, and is a trained model available?
  GET  /c4/features   the features the model expects, with labels and scales
  POST /c4/predict    one project's features -> probability, explanation, forecast, recommendations
  GET  /c4/demo       the same for a held-out synthetic project (for the Predict screen before real data)

Academic marks are not accepted here: the prediction never uses them.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..common import config
from ..synthetic.feature_spec import FEATURES
from . import service

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
