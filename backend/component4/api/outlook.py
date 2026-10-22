"""The supervisor's status dots, suggested from each group's prediction (the supervisor's own choice wins)."""
from ..common import config
from ..m1_feature_integration.team_sources import team_project
from ..m6_improvement import history
from ..synthetic.feature_spec import FEATURES
from . import service


def status(probability: float, features: dict, rules: dict | None = None) -> str:
    rules = rules or config.STATUS_RULES
    flagged = any((features.get(name) or 0) >= at_least for name, at_least in rules["attention_when"].items())
    if flagged or probability < rules["fine_at"]:
        return "attention"
    return "good" if probability >= rules["good_at"] else "fine"


def team_outlook(conn, team_id: str, bundle: dict) -> dict:
    project = team_project(conn, team_id, allow_network=False)
    signals = len(FEATURES) - len(project["missing"])
    if signals < config.API["min_team_signals"]:
        return {"team_id": team_id, "enough_data": False, "status": None, "probability": None}
    result = service.quick(project["features"], bundle)
    history.record(conn, team_id, result, project["features"])
    return {"team_id": team_id, "enough_data": True, "probability": result["probability"],
            "status": status(result["probability"], project["features"])}
