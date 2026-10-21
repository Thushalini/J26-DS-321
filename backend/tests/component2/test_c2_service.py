"""The API's results (api/service.py): stage shares, lagging stages, daily counts, drift, members and roles."""
import pytest

from component2.api import service
from component2.config import STAGES
from component2.db import save_drift, save_labels, save_roles

TEAM = "demo__service"
A, B, C = "M-aaaaaa", "M-bbbbbb", "M-cccccc"


def work(add):
    add(TEAM, [("old", A, "commit", "2026-08-01T10:00:00Z", "old cleaning", "data_cleaning"),   # outside 30 days
               ("w1", A, "commit", "2026-10-15T10:00:00Z", "clean", "data_cleaning"),
               ("w2", A, "commit", "2026-10-15T12:00:00Z", "clean more", "data_cleaning"),
               ("w3", B, "notebook_cell", "2026-10-18T10:00:00Z", "plot", "eda"),
               ("w4", B, "commit", "2026-10-20T10:00:00Z", "fit model", "model_development"),
               ("w5", B, "commit", "2026-10-20T11:00:00Z", "merge branch", None),             # unlabelled
               ("ch1", C, "chat", "2026-10-21T09:00:00Z", "I'll train the model", "model_development"),
               ("ch2", C, "chat", "2026-10-21T09:05:00Z", "model ready soon", "model_development")])


def test_list_teams(conn, add):
    work(add)
    assert service.list_teams(conn) == [{"team_id": TEAM, "activities": 8,
                                         "first_activity": "2026-08-01T10:00:00Z",
                                         "last_activity": "2026-10-21T09:05:00Z"}]


def test_stage_breakdown_uses_recent_code_work_only(conn, add):
    work(add)
    window, shares = service.stage_breakdown(conn, TEAM, 30)
    assert window == {"start": "2026-09-20T11:00:00Z", "end": "2026-10-20T11:00:00Z"}   # ends at last commit, not chat
    assert shares["data_cleaning"] == 0.5 and shares["eda"] == 0.25 and shares["model_development"] == 0.25
    assert set(shares) == set(STAGES) and sum(shares.values()) == pytest.approx(1.0)


def test_trained_model_label_wins_over_keyword_baseline(conn, add):
    work(add)
    save_labels(conn, "baseline", [("w3", "documentation", 0.4)])
    save_labels(conn, "baseline", [("w5", "documentation", 0.4)])      # only baseline -> baseline is used
    labels = service._labels_for_team(conn, TEAM)
    assert labels["w3"] == "eda" and labels["w5"] == "documentation"


def test_lagging_stages_are_below_five_percent(conn, add):
    work(add)
    summary = service.team_summary(conn, TEAM, 30)
    assert set(summary["lagging_stages"]) == {"feature_engineering", "model_evaluation",
                                              "deployment_mlops", "documentation"}


def test_no_code_work_means_no_summary(conn, add):
    add("chat_only", [("x", A, "chat", "2026-10-21T09:00:00Z", "hello", None)])
    assert service.stage_breakdown(conn, "chat_only", 30) == (None, {})
    assert service.team_summary(conn, "chat_only") is None
    assert service.daily_stage_counts(conn, "chat_only", 30) == []


def test_daily_counts_put_unlabelled_work_under_other(conn, add):
    work(add)
    days = {d["date"]: d for d in service.daily_stage_counts(conn, TEAM, 30)}
    assert list(days) == ["2026-10-15", "2026-10-18", "2026-10-20"]
    assert days["2026-10-15"]["data_cleaning"] == 2
    assert days["2026-10-20"]["model_development"] == 1 and days["2026-10-20"]["other"] == 1


def drift_row(window_end, dim, score, severity):
    return {"team_id": TEAM, "window_end": window_end, "dimension": dim, "planned": "x", "seen_top": "y",
            "drift_score": score, "severity": severity, "explanation": ""}


def test_drift_keeps_the_latest_window_and_worst_severity(conn):
    save_drift(conn, [drift_row("2026-10-01T00:00:00Z", "algorithm", 0.9, "significant"),
                      drift_row("2026-10-20T00:00:00Z", "algorithm", 0.3, "minor"),
                      drift_row("2026-10-20T00:00:00Z", "metric", 0.5, "moderate"),
                      drift_row("2026-10-20T00:00:00Z", "target", None, "no_plan")])
    out = service.drift(conn, TEAM)
    by_dim = {d["dimension"]: d for d in out["dimensions"]}
    assert by_dim["algorithm"]["drift_score"] == 0.3              # the old 0.9 window is replaced
    assert out["overall_severity"] == "moderate"


def test_drift_without_graded_dimensions_is_unknown(conn):
    save_drift(conn, [drift_row("2026-10-20T00:00:00Z", "target", None, "no_plan")])
    assert service.drift(conn, TEAM)["overall_severity"] == "unknown"
    assert service.drift(conn, "nobody") == {"overall_severity": "unknown", "dimensions": []}


def test_members_role_history_is_trimmed_and_roster_is_listed(conn):
    save_roles(conn, [{"team_id": TEAM, "member_id": A, "week": f"2026-W{w:02d}", "stage_mix": {"eda": 1.0},
                       "chat": {}, "dominant_stage": "eda"} for w in range(20, 40)])
    members = {m["member_id"]: m for m in service.members(conn, TEAM, roster=[A, C])}
    assert len(members[A]["role_trend"]) == service.TREND_WEEKS
    assert members[A]["role_trend"][-1]["week"] == "2026-W39"
    assert members[C] == {"member_id": C, "stage_mix": {}, "chat": {}, "dominant_role": None, "role_trend": []}


def test_member_roles_count_chat_at_half_weight(conn, add):
    work(add)
    roles = {m["member_id"]: m for m in service.member_roles(conn, TEAM, "30d")["members"]}
    assert roles[A]["code_items"] == 2 and roles[A]["main_roles"] == ["data_cleaning"]
    assert roles[B]["combined_mix"]["eda"] == 0.5 and roles[B]["main_roles"] == ["eda", "model_development"]
    assert roles[C]["code_items"] == 0 and roles[C]["chat_items"] == 2
    assert roles[C]["chat_mix"]["model_development"] == 1.0 and roles[C]["evidence"] == "low"


def test_chat_about_a_stage_counts_half_as_much_as_doing_it(conn, add):
    add("mixed", [("d1", A, "commit", "2026-10-20T10:00:00Z", "plots", "eda"),
                  ("d2", A, "chat", "2026-10-20T11:00:00Z", "training soon", "model_development"),
                  ("d3", A, "chat", "2026-10-20T12:00:00Z", "still training", "model_development")])
    roles = service.member_roles(conn, "mixed", "7d")
    assert roles["chat_weight"] == 0.5
    mix = roles["members"][0]["combined_mix"]
    assert mix["eda"] == 0.5 and mix["model_development"] == 0.5      # 1 commit = 2 chat messages


def test_member_roles_window_and_evidence(conn, add):
    work(add)
    week = {m["member_id"]: m for m in service.member_roles(conn, TEAM, "3d", roster=["M-dddddd"])["members"]}
    assert week[A]["evidence"] == "none" and week[A]["last_active"] == "2026-10-15T12:00:00Z"   # active, not recently
    assert week["M-dddddd"]["evidence"] == "none" and week["M-dddddd"]["last_active"] is None
    everything = {m["member_id"]: m for m in service.member_roles(conn, TEAM, "all")["members"]}
    assert everything[A]["code_items"] == 3 and everything[A]["evidence"] == "ok"


def test_member_roles_for_an_unknown_team(conn):
    assert service.member_roles(conn, "nobody") is None
