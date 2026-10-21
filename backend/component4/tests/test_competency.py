import pytest

from component4.common import config
from component4.m6_improvement.competency.scoring import competency, load_module_stages, suggest_helpers
from component4.m6_improvement.competency.sheets import SheetError, parse_text

SHEET = """Student Performance Profile
Registration No: IT00000000
Full Name: TEST STUDENT
Subject Code Subject Name Credits Attempt Grade
IT2110 Probability & Statistics 3 1 A
IT3071 Machine Learning and Optimization Methods 4 1 C
IT3071 Machine Learning and Optimization Methods 4 2 A-
IT3031 Database Systems and Data-Driven Application 4 1 F(D+)
IT1040 Communication Skills 3 1 B
"""


def test_reads_modules_and_keeps_only_the_latest_attempt():
    rows = {r["code"]: r for r in parse_text(SHEET)}
    assert rows["IT3071"]["grade"] == "A-" and rows["IT3031"]["grade"] == "F"
    assert set(rows) == {"IT2110", "IT3071", "IT3031", "IT1040"}


def test_no_name_or_registration_number_is_kept():
    for row in parse_text(SHEET):
        assert set(row) == {"code", "name", "credits", "grade"}


def test_a_page_without_results_is_rejected():
    with pytest.raises(SheetError):
        parse_text("Student Performance Profile\nnothing here")


def test_competency_is_between_zero_and_one_per_stage():
    scores = competency(parse_text(SHEET))
    assert scores and all(0 <= s["score"] <= 1 for s in scores.values())
    assert set(scores) <= set(config.STAGE_LABELS)


def test_better_grades_mean_higher_competency():
    strong = competency(parse_text(SHEET.replace("4 2 A-", "4 2 A+")))
    weak = competency(parse_text(SHEET.replace("4 2 A-", "4 2 C")))
    assert strong["model_development"]["score"] > weak["model_development"]["score"]


def test_suggests_the_strongest_member_without_showing_marks():
    members = {"a": {"name": "Ann", "competency": {"eda": {"score": 0.9, "evidence": 5}}},
               "b": {"name": "Ben", "competency": {"eda": {"score": 0.6, "evidence": 5}}}}
    (only,) = suggest_helpers(["eda"], members, config.STAGE_LABELS)
    assert only["member_name"] == "Ann"
    assert "score" not in only and "0.9" not in only["reason"]


def test_too_little_related_coursework_is_not_treated_as_weak():
    members = {"a": {"name": "Ann", "competency": {"eda": {"score": 1.0, "evidence": 0.5}}}}
    (only,) = suggest_helpers(["eda"], members, config.STAGE_LABELS)
    assert only["member_id"] is None


def test_every_mapped_stage_is_a_component2_stage():
    for module in load_module_stages()["modules"].values():
        assert set(module["stages"]) <= set(config.STAGE_LABELS)


def test_weak_stages_combine_their_sources():
    from component4.m6_improvement.weak_stages import weak_stages
    found = {w["stage"]: w["sources"] for w in weak_stages(
        ["documentation"], ["data_leakage"], [{"stage": "data_cleaning"}, {"stage": None}])}
    assert found["documentation"] == ["behavior"]
    assert found["data_cleaning"] == ["issues", "prediction"]
    assert found["feature_engineering"] == ["issues"]


def test_weak_readiness_areas_become_weak_stages():
    from component4.m6_improvement.weak_stages import weak_stages
    readiness = {"areas": [{"stage": "documentation", "score": 0}, {"stage": "deployment_mlops", "score": 100}]}
    found = {w["stage"]: w["sources"] for w in weak_stages([], [], [], readiness)}
    assert found == {"documentation": ["readiness"]}
