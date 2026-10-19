from datetime import datetime, timezone

from component4.m1_feature_integration.c3_adapter import project_features
from component4.synthetic.feature_spec import FEATURES

NOW = datetime(2026, 10, 18, tzinfo=timezone.utc)


def run(level, severity, anomaly=0.5, **extra):
    return {"severity_level": level, "severity_score": severity, "anomaly_score": anomaly, **extra}


def test_counts_medium_and_high_as_issues():
    result = project_features([run("LOW", 0.2), run("MEDIUM", 0.5), run("HIGH", 0.8), run("LOW", 0.1)])
    assert result["features"]["issue_count"] == 2
    assert result["features"]["issue_share"] == 0.5


def test_reports_worst_and_average_severity():
    features = project_features([run("LOW", 0.2), run("HIGH", 0.8)])["features"]
    assert features["max_severity"] == 0.8 and features["mean_severity"] == 0.5


def test_only_unresolved_high_issues_are_open():
    runs = [run("HIGH", 0.9), run("HIGH", 0.8, resolved=True), run("MEDIUM", 0.5)]
    assert project_features(runs)["features"]["open_high_issues"] == 1


def test_is_issue_field_wins_when_present():
    assert project_features([run("LOW", 0.2, is_issue=True)])["features"]["issue_count"] == 1


def test_filters_to_one_team_but_keeps_runs_without_a_team():
    runs = [run("HIGH", 0.9, team_id="a"), run("HIGH", 0.9, team_id="b"), run("HIGH", 0.9)]
    assert project_features(runs, team_id="a")["context"]["runs"] == 2


def test_recent_window_needs_every_run_to_have_a_time():
    timed = [run("HIGH", 0.9, run_ended_at="2026-10-15T10:00:00Z"),
             run("HIGH", 0.9, run_ended_at="2026-09-01T10:00:00Z")]
    context = project_features(timed, now=NOW)["context"]
    assert context["recent_window_available"] and context["recent_issue_count"] == 1

    untimed = project_features([*timed, run("HIGH", 0.9)], now=NOW)["context"]
    assert not untimed["recent_window_available"] and untimed["recent_issue_count"] is None


def test_no_runs_gives_zeros():
    result = project_features([])
    assert all(value == 0 for value in result["features"].values())
    assert result["context"]["runs"] == 0


def test_produces_exactly_the_c3_features_in_the_schema():
    expected = {spec.name for spec in FEATURES if spec.source == "C3"}
    assert set(project_features([run("HIGH", 0.9)])["features"]) == expected
