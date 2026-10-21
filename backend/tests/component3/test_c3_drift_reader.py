"""Tests for Component 3's drift reader: one drift number per team from Component 2's drift_results rows."""
from component3.collector.drift_reader import combine


def test_latest_window_of_each_dimension_then_the_largest_score():
    rows = [("algorithm", "2026-10-10T10:00:00Z", 0.30, "moderate"),
            ("algorithm", "2026-10-17T10:00:00Z", 0.70, "significant"),
            ("metric", "2026-10-17T10:00:00Z", 0.43, "moderate")]
    out = combine(rows)
    assert out["drift_severity_score"] == 0.70 and out["source"] == "component2"
    assert out["dimensions"]["algorithm"]["window_end"] == "2026-10-17T10:00:00Z"


def test_dimensions_without_a_score_are_ignored():
    rows = [("algorithm", "2026-10-17T10:00:00Z", None, "no_plan"), ("metric", "2026-10-17T10:00:00Z", 0.43, "moderate")]
    assert combine(rows)["drift_severity_score"] == 0.43


def test_no_score_at_all_gives_none():
    out = combine([("algorithm", "2026-10-17T10:00:00Z", None, "no_plan")])
    assert out["drift_severity_score"] is None and out["source"] == "no_score"
