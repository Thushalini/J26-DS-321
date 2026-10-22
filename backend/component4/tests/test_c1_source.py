from component4.common import config
from component4.m1_feature_integration import c1_source

ORDER = config.TIMELINE["pace_order"]


def test_an_even_pace_decides_which_stages_a_milestone_expects():
    assert c1_source.expected_stages("2026-01-02", "2026-01-01", "2026-07-01") == ORDER[:1]
    assert c1_source.expected_stages("2026-04-01", "2026-01-01", "2026-07-01") == ORDER[:3]
    assert c1_source.expected_stages("2026-07-01", "2026-01-01", "2026-07-01") == ORDER


def test_milestones_keep_their_own_stages_skip_undated_ones_and_add_the_deadline():
    snap = {"deadline": "2026-07-01", "milestones": [
        {"name": "Demo", "date": "2026-04-01"}, {"name": "Viva", "date": None, "week": 12},
        {"name": "PP1", "date": "2026-02-01", "stages": ["eda"]}]}
    found = c1_source.milestones(snap, "2026-01-01")
    assert [(m["name"], m["stages"]) for m in found] == [
        ("Demo", ORDER[:3]), ("PP1", ["eda"]), (config.TIMELINE["deadline_label"], ORDER)]


def test_without_a_start_date_only_the_deadline_expects_stages():
    found = c1_source.milestones({"deadline": "2026-07-01", "milestones": [{"name": "Demo", "date": "2026-04-01"}]}, None)
    assert found[0]["stages"] == [] and found[1]["stages"] == ORDER


def test_project_start_prefers_the_repo_then_the_first_commit_then_the_group():
    assert c1_source.project_start("2026-07-01T09:00:00Z", "2026-07-10T00:00:00Z", "2026-08-01") == "2026-07-01"
    assert c1_source.project_start(None, "2026-07-10T00:00:00Z", "2026-08-01T00:00:00Z") == "2026-07-10"
    assert c1_source.project_start(None, None, "2026-08-01T00:00:00Z") == "2026-08-01"
    assert c1_source.project_start(None, None, None) is None
