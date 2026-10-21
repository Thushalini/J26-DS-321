from component4.m1_feature_integration.timeline import adherence

DAYS = [{"date": "2026-11-01", "data_cleaning": 2, "eda": 1}, {"date": "2026-11-20", "data_cleaning": 2, "eda": 3}]
MILESTONES = [
    {"name": "PP1", "date": "2026-12-01", "stages": ["data_cleaning", "eda"]},
    {"name": "PP2", "date": "2027-02-01", "stages": ["model_development"]},
    {"name": "Viva", "date": "2027-04-01", "stages": ["documentation"]},
]


def test_nothing_due_yet_means_not_measured():
    assert adherence(MILESTONES, DAYS, [], today="2026-11-15") is None


def test_a_milestone_is_met_only_when_its_stages_show_enough_work_by_its_date():
    assert adherence(MILESTONES[:1], DAYS, [], today="2026-12-05") == 1.0
    assert adherence(MILESTONES[:1], DAYS[:1], [], today="2026-12-05") == 0.0


def test_component3_runs_count_for_model_stages():
    runs = [{"created_at": f"2027-01-1{i}T10:00:00Z"} for i in range(3)]
    assert adherence(MILESTONES[:2], DAYS, runs, today="2027-02-02") == 1.0
    assert adherence(MILESTONES[:2], DAYS, runs[:1], today="2027-02-02") == 0.5


def test_milestones_without_stages_are_only_reminders():
    assert adherence([{"name": "Viva", "date": "2026-01-01", "stages": []}], DAYS, [], today="2027-01-01") is None
