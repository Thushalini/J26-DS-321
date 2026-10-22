"""Tests for component1/pipeline.py (the connected flow for one message).

A pretend language model and a small model trained on simulated data are used, so no
internet and no saved model file are needed. All values are invented.
"""
import json

from sklearn.ensemble import RandomForestClassifier

from component1.agent.signals import CONFLICT, GAP, SAVED
from component1.common.events import EventStore
from component1.common.schema import FieldName, FieldStatus
from component1.pipeline import process_message
from component1.state.features import FEATURE_ORDER
from component1.state.simulate import FEATURE_NAMES, generate_dataset


def small_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


BUNDLE = small_bundle()


def says(field, value, span, hedged=False):
    item = {"field": field, "value": value, "span": span, "hedged": hedged}
    return lambda prompt: json.dumps({"items": [item]})


def says_nothing(prompt):
    return json.dumps({"items": []})


def run(store, llm, message_id, speaker, text):
    return process_message(store, BUNDLE, llm, message_id, speaker, text)


TARGET = "target_variable"


def test_a_hedged_first_mention_is_recorded_as_uncertain():
    store = EventStore("p1")
    r = run(store, says(TARGET, "tourist arrivals", "predict tourist arrivals", hedged=True),
            "M1", "Member_A", "I think we should predict tourist arrivals.")
    assert r.statuses[TARGET][0] == "uncertain"
    f = store.state.fields[FieldName.TARGET_VARIABLE]
    assert f.status == FieldStatus.UNCERTAIN and f.value == "tourist arrivals"


def test_a_second_person_with_a_different_value_makes_it_conflicting():
    store = EventStore("p1")
    run(store, says(TARGET, "tourist arrivals", "tourist arrivals"), "M1", "Member_A", "predict tourist arrivals")
    r = run(store, says(TARGET, "hotel occupancy", "hotel occupancy"), "M2", "Member_C", "the target is hotel occupancy")
    assert r.statuses[TARGET][0] == "conflicting"
    conflicts = [s for s in r.signals if s.kind == CONFLICT]
    assert len(conflicts) == 1
    assert set(conflicts[0].values) == {"tourist arrivals", "hotel occupancy"}


def test_a_second_person_restating_the_value_confirms_it():
    store = EventStore("p1")
    run(store, says("algorithm", "XGBoost", "XGBoost"), "M1", "Member_B", "Let's use XGBoost")
    r = run(store, says("algorithm", "xgboost", "xgboost"), "M2", "Member_C", "yes, xgboost it is")
    assert r.statuses["algorithm"][0] == "confirmed"
    assert not [s for s in r.signals if s.kind == CONFLICT]


def test_the_same_person_repeating_is_still_only_known():
    store = EventStore("p1")
    run(store, says("algorithm", "XGBoost", "XGBoost"), "M1", "Member_B", "Let's use XGBoost")
    r = run(store, says("algorithm", "XGBoost", "XGBoost"), "M2", "Member_B", "XGBoost for sure")
    assert r.statuses["algorithm"][0] == "known"


def test_nothing_extracted_changes_nothing_but_gaps_are_still_reported():
    store = EventStore("p1")
    r = run(store, says_nothing, "M1", "Member_C", "What time is the meeting?")
    assert store.state.version == 0
    assert r.statuses == {} and r.saved == []
    assert [s.field for s in r.signals if s.kind == GAP] == ["problem"]
    assert not [s for s in r.signals if s.kind == SAVED]


def test_the_saved_signal_names_the_speaker_and_the_status():
    store = EventStore("p1")
    r = run(store, says(TARGET, "tourist arrivals", "predict tourist arrivals", hedged=True),
            "M1", "Member_A", "I think we should predict tourist arrivals.")
    (saved,) = [s for s in r.signals if s.kind == SAVED]
    assert saved.source == "Member_A" and saved.detail == "marked uncertain"
    assert saved.values == ("tourist arrivals",)


def test_gap_signals_skip_filled_fields_and_follow_importance_order(monkeypatch):
    monkeypatch.setenv("C1_ACTIVE", "off")           # the old fixed order (importance weight only)
    store = EventStore("p1")
    r = run(store, says("problem", "forecast arrivals", "forecast arrivals"), "M1", "Member_A", "we need to forecast arrivals")
    assert [s.field for s in r.signals if s.kind == GAP] == ["objective", "dataset", "domain", "timeline"]
    assert all(s.priority == 0 for s in r.signals)


def test_with_priorities_a_field_others_depend_on_is_asked_first():
    store = EventStore("p1")
    r = run(store, says("problem", "forecast arrivals", "forecast arrivals"), "M1", "Member_A", "we need to forecast arrivals")
    gaps = [s for s in r.signals if s.kind == GAP]
    assert [s.field for s in gaps] == ["dataset", "objective", "domain", "timeline"]   # most other fields depend on the dataset
    assert gaps[0].priority > gaps[1].priority > 0 and "depend on it" in gaps[0].detail


def test_the_fields_confidence_is_the_models_probability():
    store = EventStore("p1")
    r = run(store, says("algorithm", "XGBoost", "XGBoost"), "M1", "Member_B", "Let's use XGBoost")
    confidence = store.state.fields[FieldName.ALGORITHM].confidence
    assert confidence == r.statuses["algorithm"][1] and 0.0 < confidence <= 1.0


def test_recording_a_status_does_not_copy_the_evidence():
    store = EventStore("p1")
    run(store, says("algorithm", "XGBoost", "XGBoost"), "M1", "Member_B", "Let's use XGBoost")
    run(store, says("algorithm", "xgboost", "xgboost"), "M2", "Member_C", "yes, xgboost it is")
    assert len(store.state.fields[FieldName.ALGORITHM].provenance) == 2   # two mentions, not four


def test_replaying_the_events_still_gives_the_same_blueprint():
    store = EventStore("p1")
    run(store, says(TARGET, "tourist arrivals", "tourist arrivals"), "M1", "Member_A", "predict tourist arrivals")
    run(store, says(TARGET, "hotel occupancy", "hotel occupancy"), "M2", "Member_C", "the target is hotel occupancy")
    assert store.rebuild().model_dump(mode="json") == store.state.model_dump(mode="json")