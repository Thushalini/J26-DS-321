import pytest

from component4.common import config
from component4.m2_prediction.data import split
from component4.m2_prediction.train import train_and_evaluate
from component4.m5_forecast.cone import cone_checks, forecast
from component4.synthetic import generator

SMALL = {"simulations": 100}


@pytest.fixture(scope="module")
def setup():
    frame = generator.generate(500)
    _, model = train_and_evaluate(frame, search=False)
    project = split(frame)["test"][0].iloc[0]
    return model, project, model.predict_proba(project.to_frame().T.astype(float))[:, 1][0]


def test_one_point_per_week_in_the_frontend_shape(setup):
    model, project, _ = setup
    points = forecast(model, project, settings=SMALL)
    assert [p["week"] for p in points] == list(range(config.FORECAST["weeks"] + 1))
    assert set(points[0]) == {"week", *(f"p{p}" for p in config.FORECAST["percentiles"])}


def test_cone_passes_its_checks(setup):
    model, project, now = setup
    assert all(c["passed"] for c in cone_checks(forecast(model, project, settings=SMALL), now))


def test_same_seed_same_cone(setup):
    model, project, _ = setup
    assert forecast(model, project, seed=3, settings=SMALL) == forecast(model, project, seed=3, settings=SMALL)


def test_improvement_trend_raises_the_outlook(setup):
    model, project, _ = setup
    plain = forecast(model, project, settings=SMALL)[-1]["p50"]
    improving = forecast(model, project, trend={"drift_score": -0.05, "timeline_adherence": 0.05}, settings=SMALL)[-1]["p50"]
    assert improving > plain
