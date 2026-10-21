"""Tests for the timeline field: finding it, turning it into dates, milestones, and the plan.

All dates and messages are invented. Fixed dates are used, so the tests never depend on today.
"""
import json
from datetime import date, datetime, timezone

import pytest
from sklearn.ensemble import RandomForestClassifier

from component1.common.events import EventStore
from component1.common.schema import BASE_WEIGHT, DEPENDS_ON, DecisionStatus, FieldName, FieldStatus
from component1.documents.chunk import Chunk
from component1.documents.extract import extract_document
from component1.extraction.extractor import extract_items
from component1.extraction.hybrid import make_hybrid_llm
from component1.extraction.rules import rule_items
from component1.pipeline import process_message
from component1.plan import plan_from_events
from component1.state.features import FEATURE_ORDER
from component1.state.simulate import FEATURE_NAMES, generate_dataset
from component1.timeline.dates import duration_days, parse_date, resolve, resolve_milestone
from component1.timeline.find import find_timeline, standard_timeline_value
from component1.timeline.milestones import milestone_list, record_milestone

RULES = make_hybrid_llm(None, mode="rules")
MON_19_OCT_2026 = date(2026, 10, 19)          # a Monday


def small_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


BUNDLE = small_bundle()


def timeline_of(text):
    return [(i.value, i.hedged) for i in rule_items(text) if i.field == FieldName.TIMELINE]


# ------------------------------------------------------------- the field
def test_timeline_is_a_blueprint_field_asked_after_the_core_plan():
    assert FieldName.TIMELINE in DEPENDS_ON and DEPENDS_ON[FieldName.TIMELINE] == [FieldName.PROBLEM]
    assert BASE_WEIGHT[FieldName.EVALUATION_METRIC] > BASE_WEIGHT[FieldName.TIMELINE] > BASE_WEIGHT[FieldName.RESOURCES]


# ----------------------------------------------------------------- finding
@pytest.mark.parametrize("text, value", [
    ("We have 8 weeks for this project.", "8 weeks"),
    ("It's an eight-week project.", "8 weeks"),
    ("The project lasts 14 weeks.", "14 weeks"),
    ("Only two months left.", "2 months"),
    ("Submission is on 15 December.", "deadline 15 december"),
    ("The deadline is 15/12/2026.", "deadline 2026-12-15"),
    ("We must submit by December 1st 2026.", "deadline 2026-12-01"),
    ("Final report due in 3 weeks.", "deadline in 3 weeks"),
    ("The deadline is next Friday.", "deadline next friday"),
])
def test_the_overall_timeline_is_found_in_standard_form(text, value):
    assert timeline_of(text) == [(value, False)]


@pytest.mark.parametrize("text", [
    "See you in 2 weeks!", "Meeting tomorrow at 10.", "I'll be away for 2 weeks.", "It took me a week to fix.",
    "Born on 15 December.", "Let's use XGBoost.",
])
def test_time_words_without_a_planning_cue_are_ignored(text):
    assert timeline_of(text) == []


def test_a_denied_date_is_skipped():
    assert timeline_of("The deadline is not 15 Dec, it's 20 Dec.") == [("deadline 20 dec", False)]


def test_milestones_are_separate_from_the_deadline():
    overall, milestones = find_timeline("The deadline is 15/12/2026 and the demo is on 20/11/2026.")
    assert [f.value for f in overall] == ["deadline 2026-12-15"]
    assert [f.value for f in milestones] == ["milestone: Demo, 2026-11-20"]
    sentence = "The deadline is 15/12/2026 and the demo is on 20/11/2026."
    assert all(f.span in sentence for f in overall + milestones)


def test_teaching_weeks_and_named_steps_become_milestones():
    _, milestones = find_timeline("Mid evaluation in week 10, presentation next Friday.")
    assert [f.value for f in milestones] == ["milestone: Mid evaluation, week 10", "milestone: Presentation, next friday"]


def test_unsure_timelines_are_hedged():
    assert timeline_of("Maybe we have 10 weeks?") == [("10 weeks", True)]


@pytest.mark.parametrize("value, standard", [
    ("eight weeks", "8 weeks"), ("8 Weeks", "8 weeks"), ("1 month", "1 month"), ("15 December 2026", "deadline 2026-12-15"),
    ("deadline 15/12/2026", "deadline 2026-12-15"), ("milestone: Demo, 20 November", "milestone: Demo, 20 November"),
    ("end of semester", "end of semester"),
])
def test_language_model_values_are_brought_into_standard_form(value, standard):
    assert standard_timeline_value(value) == standard


def test_the_extractor_standardises_timeline_values_from_the_language_model():
    reply = json.dumps({"items": [{"field": "timeline", "value": "eight weeks", "span": "eight weeks"}]})
    (item,) = extract_items("We have roughly eight weeks.", lambda prompt: reply)
    assert item.value == "8 weeks"


# ------------------------------------------------------------------- dates
def test_slash_dates_are_day_first_unless_told_otherwise(monkeypatch):
    assert parse_date("05/06/2026") == date(2026, 6, 5)
    monkeypatch.setenv("C1_DATES_MONTH_FIRST", "1")
    assert parse_date("05/06/2026") == date(2026, 5, 6)


@pytest.mark.parametrize("text, expected", [
    ("2026-12-15", date(2026, 12, 15)),
    ("15 December", date(2026, 12, 15)),               # next 15 December after 19 Oct 2026
    ("15 march", date(2027, 3, 15)),                   # already past this year -> next year
    ("Dec 1st, 2026", date(2026, 12, 1)),
    ("next friday", date(2026, 10, 23)),
    ("next monday", date(2026, 10, 26)),               # "next" on a Monday is a week later
    ("tomorrow", date(2026, 10, 20)),
    ("in 3 weeks", date(2026, 11, 9)),
    ("31/02/2026", None),                              # no such date
    ("week 10", None),
])
def test_dates_are_worked_out_from_when_they_were_said(text, expected):
    assert parse_date(text, MON_19_OCT_2026) == expected


def test_a_length_ends_that_long_after_it_was_said_and_days_left_are_counted():
    r = resolve("8 weeks", said_on=datetime(2026, 10, 1, tzinfo=timezone.utc), today=date(2026, 10, 19))
    assert (r["kind"], r["duration_days"], r["deadline"], r["days_left"]) == ("duration", 56, "2026-11-26", 38)


def test_deadlines_teaching_weeks_and_plain_text_resolve():
    assert resolve("deadline 15 december", "2026-10-19T10:00:00Z", today=date(2026, 12, 20))["days_left"] == -5
    assert resolve("deadline week 12", None)["kind"] == "week" and resolve("deadline week 12", None)["week"] == 12
    assert resolve("end of semester", None)["kind"] == "text"
    assert resolve(None) is None and resolve("milestone: Demo, 2026-11-20") is None
    assert duration_days("two months") == 60 and duration_days("soon") is None


def test_milestones_resolve_to_dates_or_weeks():
    assert resolve_milestone("milestone: Demo, next friday", said_on=MON_19_OCT_2026) == \
        {"name": "Demo", "when": "next friday", "date": "2026-10-23", "week": None}
    assert resolve_milestone("milestone: Mid evaluation, week 10")["week"] == 10


# -------------------------------------------------------------- milestones
def test_a_milestone_is_a_decision_and_moving_it_supersedes_the_old_date():
    store = EventStore("t")
    assert record_milestone(store, "milestone: Demo, 2026-11-20", "demo is on 20/11/2026", "M-1", "m1") is not None
    assert record_milestone(store, "milestone: Demo, 2026-11-20", "demo on 20/11/2026", "M-2", "m2") is None   # same
    record_milestone(store, "milestone: Demo, 2026-11-27", "demo moved to 27/11/2026", "M-1", "m3")
    decisions = list(store.state.decisions.values())
    assert [d.status for d in decisions] == [DecisionStatus.SUPERSEDED, DecisionStatus.PROPOSED]
    assert decisions[1].supersedes == decisions[0].decision_id
    (only,) = milestone_list(store.state)
    assert (only["name"], only["date"], only["proposed_by"]) == ("Demo", "2026-11-27", "M-1")


def test_milestones_from_chat_do_not_touch_the_timeline_field():
    store = EventStore("t")
    process_message(store, BUNDLE, RULES, "m1", "M-1", "The demo is on 20/11/2026.")
    assert store.state.fields[FieldName.TIMELINE].status == FieldStatus.MISSING
    assert [m["name"] for m in milestone_list(store.state)] == ["Demo"]


def test_milestones_survive_a_restart():
    store = EventStore("t")
    process_message(store, BUNDLE, RULES, "m1", "M-1", "Mid evaluation in week 10 and the demo is on 20/11/2026.")
    again = EventStore.from_events("t", store.events)
    assert [m["name"] for m in milestone_list(again.state)] == ["Demo", "Mid evaluation"]   # dated first


# ------------------------------------------------------- field behaviour
def test_two_ways_of_writing_the_same_length_agree_and_different_lengths_conflict():
    store = EventStore("t")
    process_message(store, BUNDLE, RULES, "m1", "M-1", "We have 8 weeks for this project.")
    process_message(store, BUNDLE, RULES, "m2", "M-2", "It's an eight-week project.")
    assert store.state.fields[FieldName.TIMELINE].status != FieldStatus.CONFLICTING
    process_message(store, BUNDLE, RULES, "m3", "M-3", "The project lasts 10 weeks.")
    assert store.state.fields[FieldName.TIMELINE].status == FieldStatus.CONFLICTING


def test_the_plan_shows_the_worked_out_deadline_and_the_milestones():
    store = EventStore("t")
    process_message(store, BUNDLE, RULES, "m1", "M-1", "We have 8 weeks. The demo is on 20/11/2026.")
    plan = plan_from_events("t", store.events)
    timeline = next(f for f in plan["fields"] if f["name"] == "timeline")
    assert timeline["value"] == "8 weeks" and timeline["timeline"]["duration_days"] == 56
    assert timeline["timeline"]["deadline"] is not None
    assert [(m["name"], m["date"]) for m in plan["milestones"]] == [("Demo", "2026-11-20")]
    assert plan["readiness"]["total"] == 10


def test_buddy_asks_about_the_timeline_in_plain_words():
    from component1.agent.policy import gap_text
    from component1.agent.signals import GAP, Signal
    assert "deadline" in gap_text(Signal(GAP, "timeline"))


# --------------------------------------------------------------- documents
def test_documents_suggest_the_overall_timeline_but_not_milestones():
    chunk = Chunk(1, 2, "The project lasts 14 weeks. The demo is on 20/11/2026.")
    facts = extract_document([chunk], fallback_llm=None)
    assert [(f.field, f.value) for f in facts] == [("timeline", "14 weeks")]


# ------------------------------------------- Component 4 still reads the plan
def test_component_4s_adapter_still_reads_a_ten_field_blueprint():
    pytest.importorskip("component4")
    from component4.m1_feature_integration.c1_adapter import project_features

    store = EventStore("t")
    process_message(store, BUNDLE, RULES, "m1", "M-1", "We have 8 weeks for this project.")
    out = project_features(store.state.model_dump(mode="json"))
    assert out["context"]["fields_total"] == 10
