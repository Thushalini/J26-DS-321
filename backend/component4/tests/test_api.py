import joblib
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from component4.api import router as api_router
from component4.api import service
from component4.common import config
from component4.m2_prediction.data import split
from component4.m2_prediction.train import train_and_evaluate
from component4.synthetic import generator
from component4.synthetic.feature_spec import feature_names


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    frame = generator.generate(500)
    _, model = train_and_evaluate(frame, search=False)
    parts = split(frame)
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump({"model": model, "version": "test", "background": parts["train"][0].sample(20, random_state=0),
                 "medians": parts["train"][0].median().to_dict(),
                 "demo_project": parts["test"][0].iloc[0].to_dict()}, path)
    original = config.MODEL_PATH
    config.MODEL_PATH = path
    app = FastAPI()
    app.include_router(api_router.router)
    yield TestClient(app), parts["test"][0].iloc[1].to_dict()
    config.MODEL_PATH = original
    service._bundle.clear()


def test_health_reports_the_model(client):
    api, _ = client
    assert api.get("/c4/health").json()["model_ready"] is True


def test_features_lists_the_schema(client):
    api, _ = client
    assert [f["name"] for f in api.get("/c4/features").json()] == feature_names()


def test_predict_returns_everything_the_screen_needs(client):
    api, project = client
    body = api.post("/c4/predict", json={"features": project, "recurring_causes": ["overfitting"]}).json()
    assert 0 <= body["probability"] <= 100
    assert len(body["factors"]) == len(feature_names())
    assert len(body["forecast"]) == config.FORECAST["weeks"] + 1
    assert body["imputed_features"] == [] and body["source"] == "live"
    assert all({"id", "title", "reason", "area", "expected_gain"} <= set(r) for r in body["recommendations"])


def test_contributions_add_up_to_the_probability(client):
    api, project = client
    body = api.post("/c4/predict", json={"features": project, "include_forecast": False}).json()
    total = body["base_probability"] + sum(f["contribution"] for f in body["factors"])
    assert total == pytest.approx(body["probability"], abs=0.5)


def test_missing_features_are_filled_and_reported(client):
    api, project = client
    partial = {k: v for k, v in project.items() if k not in ("novelty_score", "data_quality")}
    body = api.post("/c4/predict", json={"features": partial, "include_forecast": False}).json()
    assert sorted(body["imputed_features"]) == ["data_quality", "novelty_score"]


def test_marks_and_unknown_fields_are_rejected(client):
    api, project = client
    assert api.post("/c4/predict", json={"features": {**project, "gpa": 3.5}}).status_code == 422
    assert api.post("/c4/predict", json={"features": project, "recurring_causes": ["bad_luck"]}).status_code == 422


def test_demo_uses_a_synthetic_project(client):
    api, _ = client
    assert api.get("/c4/demo").json()["source"] == "synthetic"


def test_without_a_model_the_api_says_how_to_fix_it(tmp_path):
    service._bundle.clear()
    original = config.MODEL_PATH
    config.MODEL_PATH = tmp_path / "missing.joblib"
    try:
        app = FastAPI()
        app.include_router(api_router.router)
        api = TestClient(app)
        assert api.get("/c4/health").json()["model_ready"] is False
        assert api.get("/c4/demo").status_code == 503
    finally:
        config.MODEL_PATH = original
        service._bundle.clear()
