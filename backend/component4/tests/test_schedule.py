from datetime import date

from component4.common import config
from component4.m1_feature_integration import schedule, timeline


def test_months_are_added_without_running_past_the_end_of_a_month():
    assert schedule.add_months(date(2026, 8, 14), 5) == date(2027, 1, 14)
    assert schedule.add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)


def test_the_shared_schedule_is_counted_from_the_project_start():
    found = schedule.cohort("2026-08-14")
    assert [m["name"] for m in found] == [m["name"] for m in config.COHORT_SCHEDULE["milestones"]]
    assert [m["date"] for m in found] == ["2026-09-14", "2026-10-14", "2026-11-14", "2026-12-14",
                                         "2027-01-14", "2027-03-14", "2027-04-14"]
    assert all(m["source"] == "schedule" for m in found)


def test_planning_steps_are_not_counted():
    found = schedule.cohort("2026-08-14")
    assert timeline.adherence(found, [], None, "2026-09-01") is None
    work = [{"date": "2026-09-01", "data_cleaning": 3}]
    assert timeline.adherence(found, work, None, "2026-10-22") == 1.0
    assert timeline.adherence(found, [], None, "2026-10-22") == 0.0


def test_a_group_milestone_with_the_same_name_keeps_the_schedule_stages_but_moves_the_date():
    shared = schedule.cohort("2026-08-14")
    group = [{"name": "progress presentation 1", "date": "2026-12-20", "stages": ["eda"], "paced": True},
             {"name": "Demo", "date": "2026-11-01", "stages": ["eda"], "paced": False}]
    merged = schedule.merge(group, shared)
    pp1 = [m for m in merged if m["name"].lower() == "progress presentation 1"]
    assert len(pp1) == 1 and pp1[0]["date"] == "2026-12-20" and pp1[0]["source"] == "group"
    assert pp1[0]["stages"] == next(m["stages"] for m in shared if m["name"] == "Progress presentation 1")
    assert [m["date"] for m in merged] == sorted(m["date"] for m in merged)
    assert next(m for m in merged if m["name"] == "Demo")["stages"] == ["eda"]


def test_a_group_milestone_with_its_own_stages_keeps_them():
    shared = schedule.cohort("2026-08-14")
    merged = schedule.merge([{"name": "Final viva", "date": "2027-05-01", "stages": ["eda"], "paced": False}], shared)
    assert next(m for m in merged if m["name"] == "Final viva")["stages"] == ["eda"]
