"""Role tracker: weekly stage mix, chat measures, smoothing and lasting role shifts (MiniLM replaced)."""
import json

import pytest

from component2.roles import role_tracker
from component2.roles.role_tracker import ema, find_shifts, iso_week, weekly_chat_measures, weekly_stage_mix

A, B = "M-aaaaaa", "M-bbbbbb"


@pytest.fixture(autouse=True)
def no_model(monkeypatch, fake_embed):
    monkeypatch.setattr(role_tracker, "embed", fake_embed)


def test_iso_week():
    assert iso_week("2026-10-21T09:00:00Z") == "2026-W43"
    assert iso_week("2026-01-01T00:00:00Z") == "2026-W01"
    assert iso_week("2025-12-29T00:00:00Z") == "2026-W01"       # ISO weeks can start in the old year


def test_weekly_stage_mix_shares_per_member():
    labels = {"1": "eda", "2": "eda", "3": "model_development", "4": "other"}
    mix = weekly_stage_mix([("1", A, "W1"), ("2", A, "W1"), ("3", A, "W1"), ("4", B, "W1")], labels, [A, B])
    assert mix["W1"][A]["eda"] == pytest.approx(2 / 3)
    assert mix["W1"][A]["model_development"] == pytest.approx(1 / 3)
    assert sum(mix["W1"][B].values()) == 0                      # 'other' is not a stage


def test_chat_measures():
    rows = [("m1", A, "W1", "the model is trained", None),
            ("m2", B, "W1", "the model is trained", "m1"),          # B answers A, same meaning
            ("m3", B, "W1", "lunch later", "m2")]                   # answering yourself doesn't count
    talk = weekly_chat_measures(rows, [A, B])["W1"]
    assert talk[A]["participation"] == pytest.approx(1 / 3)
    assert talk[B]["participation"] == pytest.approx(2 / 3)
    assert talk[B]["responsiveness"] == pytest.approx(0.5)          # one related reply out of 2 messages
    assert talk[A]["social_impact"] == pytest.approx(1.0)
    assert weekly_chat_measures([], [A]) == {}


def test_ema_gives_half_weight_to_the_new_week():
    assert ema(None, {"eda": 1.0}) == {"eda": 1.0}
    assert ema({"eda": 1.0, "documentation": 0.0}, {"eda": 0.0, "documentation": 1.0}) == \
        {"eda": 0.5, "documentation": 0.5}


def test_a_shift_must_last_two_weeks():
    lasting = [("W1", "eda"), ("W2", "eda"), ("W3", "model_development"), ("W4", "model_development")]
    assert find_shifts(lasting) == [("W3", "eda", "model_development")]
    blip = [("W1", "eda"), ("W2", "eda"), ("W3", "documentation"), ("W4", "eda"), ("W5", "eda")]
    assert find_shifts(blip) == []
    assert find_shifts([("W1", None), ("W2", "eda"), ("W3", "eda")]) == []   # first role is not a shift


def test_track_team_smooths_and_keeps_the_role_in_quiet_weeks(conn, add):
    team = "demo__roles"
    add(team, [("a1", A, "commit", "2026-10-05T10:00:00Z", "eda work", "eda"),              # W41
               ("a2", A, "commit", "2026-10-12T10:00:00Z", "train model", "model_development"),  # W42
               ("a3", A, "commit", "2026-10-13T10:00:00Z", "train more", "model_development"),
               ("b1", B, "chat", "2026-10-20T10:00:00Z", "hello team", None)],                 # W43, chat only
        meta={"b1": {"reply_to": None}})
    rows = role_tracker.track_team(conn, team, role_tracker.stage_labels(conn))
    a = {r["week"]: r for r in rows if r["member_id"] == A}
    assert a["2026-W41"]["dominant_stage"] == "eda"
    assert a["2026-W42"]["stage_mix"]["model_development"] == 0.5 and a["2026-W42"]["stage_mix"]["eda"] == 0.5
    assert a["2026-W43"]["stage_mix"] == a["2026-W42"]["stage_mix"]     # no work that week: mix kept
    b = [r for r in rows if r["member_id"] == B]
    assert all(r["dominant_stage"] is None for r in b)                   # chat only: no stage role
    assert b[-1]["chat"]["participation"] == 0.5                        # EMA of 0 then 1
    json.dumps(rows)                                                     # saved as JSON, so it must serialise
