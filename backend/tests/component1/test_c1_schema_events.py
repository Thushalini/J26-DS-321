"""Tests for component1/common/schema.py and events.py.

These are test fixtures only. The values are invented to exercise the code.
"""
import pytest
from pydantic import ValidationError

from component1.common.events import EventStore, replay
from component1.common.schema import (
    BASE_WEIGHT,
    DEPENDS_ON,
    Blueprint,
    Decision,
    DecisionStatus,
    FieldName,
    FieldState,
    FieldStatus,
    Provenance,
    SourceType,
)


def prov(text="we should predict churn", msg="M0001"):
    return Provenance(source_type=SourceType.CHAT, source_id=msg, speaker="Member_A", text_span=text)


# ------------------------------------------------------------ schema
def test_new_blueprint_has_every_field_and_all_are_missing():
    bp = Blueprint(project_id="p1")
    assert set(bp.fields) == set(FieldName)
    assert all(f.status == FieldStatus.MISSING and f.value is None for f in bp.fields.values())
    assert bp.version == 0


def test_field_value_and_status_must_agree():
    with pytest.raises(ValidationError):
        FieldState(name=FieldName.DATASET, value="tourism data", status=FieldStatus.MISSING)
    with pytest.raises(ValidationError):
        FieldState(name=FieldName.DATASET, value=None, status=FieldStatus.KNOWN)


def test_confidence_must_be_between_0_and_1():
    with pytest.raises(ValidationError):
        FieldState(name=FieldName.DATASET, value="x", status=FieldStatus.KNOWN, confidence=1.5)


def test_provenance_needs_a_text_span():
    with pytest.raises(ValidationError):
        Provenance(source_type=SourceType.CHAT, source_id="M1", text_span="")


def test_gap_detection_config_covers_every_field_and_has_no_unknown_dependencies():
    assert set(DEPENDS_ON) == set(FieldName)
    assert set(BASE_WEIGHT) == set(FieldName)
    for deps in DEPENDS_ON.values():
        assert set(deps) <= set(FieldName)


def test_blueprint_survives_a_json_round_trip():
    store = EventStore("p1")
    store.update_field(FieldName.DATASET, "tourism dataset", FieldStatus.KNOWN, 0.8, prov(), "extractor")
    bp = store.state
    again = Blueprint.model_validate(bp.model_dump(mode="json"))
    assert again.fields[FieldName.DATASET].value == "tourism dataset"
    assert again.version == bp.version


# ------------------------------------------------------ event sourcing
def test_update_field_changes_state_and_version():
    store = EventStore("p1")
    store.update_field(FieldName.TARGET_VARIABLE, "churn", FieldStatus.KNOWN, 0.9, prov(), "extractor")
    f = store.state.fields[FieldName.TARGET_VARIABLE]
    assert f.value == "churn" and f.status == FieldStatus.KNOWN
    assert f.provenance[0].text_span == "we should predict churn"
    assert store.state.version == 1


def test_history_is_kept_when_a_value_changes():
    store = EventStore("p1")
    store.update_field(FieldName.ALGORITHM, "Random Forest", FieldStatus.KNOWN, 0.8, prov("use RF", "M1"), "extractor")
    store.update_field(FieldName.ALGORITHM, "XGBoost", FieldStatus.CONFIRMED, 0.9, prov("use XGBoost", "M2"), "extractor")
    assert store.state.fields[FieldName.ALGORITHM].value == "XGBoost"
    assert len(store.state.fields[FieldName.ALGORITHM].provenance) == 2
    second = store.events[1].payload
    assert second["old_value"] == "Random Forest" and second["new_value"] == "XGBoost"


def test_rebuilding_from_events_equals_the_current_state():
    store = EventStore("p1")
    store.update_field(FieldName.DATASET, "tourism data", FieldStatus.KNOWN, 0.7, prov(), "extractor")
    store.update_field(FieldName.DATASET, "tourism data", FieldStatus.CONFIRMED, 0.9, prov(msg="M2"), "extractor")
    store.update_scores(actor="scorer", novelty_score=0.42, feasibility={"dataset_vs_model": "pass"})
    assert store.rebuild().model_dump(mode="json") == store.state.model_dump(mode="json")
    assert replay("p1", store.events).version == 3


def test_events_cannot_be_applied_out_of_order():
    store = EventStore("p1")
    store.update_field(FieldName.DATASET, "x", FieldStatus.KNOWN, 0.5, prov(), "extractor")
    store.update_field(FieldName.DATASET, "y", FieldStatus.KNOWN, 0.5, prov(), "extractor")
    from component1.common.events import apply_event
    from component1.common.schema import Blueprint as BP
    with pytest.raises(ValueError):
        apply_event(BP(project_id="p1"), store.events[1])  # version 2 applied to version 0


def test_cannot_set_a_value_with_status_missing():
    store = EventStore("p1")
    with pytest.raises(ValueError):
        store.update_field(FieldName.DATASET, "x", FieldStatus.MISSING, 0.5, prov(), "extractor")
    assert store.state.version == 0  # nothing was stored


def test_state_returned_to_callers_is_a_copy():
    store = EventStore("p1")
    snapshot = store.state
    snapshot.fields[FieldName.DATASET].value = "tampered"
    assert store.state.fields[FieldName.DATASET].value is None


# ------------------------------------------------------------ decisions
def make_decision(did="D001"):
    return Decision(decision_id=did, field=FieldName.ALGORITHM, value="XGBoost",
                    proposed_by="Member_A", rationale="works well on tabular data")


def test_decision_goes_proposed_confirmed_finalized():
    store = EventStore("p1")
    store.record_decision(make_decision(), "intent_model")
    store.change_decision_status("D001", DecisionStatus.CONFIRMED, "intent_model", agreed_by=["Member_B"])
    store.change_decision_status("D001", DecisionStatus.FINALIZED, "intent_model")
    d = store.state.decisions["D001"]
    assert d.status == DecisionStatus.FINALIZED and d.agreed_by == ["Member_B"]


def test_invalid_decision_transitions_are_rejected():
    store = EventStore("p1")
    store.record_decision(make_decision(), "intent_model")
    store.change_decision_status("D001", DecisionStatus.REJECTED, "intent_model")
    with pytest.raises(ValueError):
        store.change_decision_status("D001", DecisionStatus.FINALIZED, "intent_model")


def test_unknown_or_duplicate_decisions_are_rejected():
    store = EventStore("p1")
    with pytest.raises(KeyError):
        store.change_decision_status("D404", DecisionStatus.CONFIRMED, "intent_model")
    store.record_decision(make_decision(), "intent_model")
    with pytest.raises(ValueError):
        store.record_decision(make_decision(), "intent_model")


def test_replay_rejects_a_hand_built_event_with_missing_status_and_a_value():
    from component1.common.events import apply_event
    from component1.common.schema import Event, EventType
    bad = Event(project_id="p1", version=1, event_type=EventType.FIELD_UPDATED, actor="db",
                payload={"field": "dataset", "old_value": None, "new_value": "x",
                         "old_status": "missing", "new_status": "missing",
                         "confidence": 0.5, "provenance": prov().model_dump(mode="json")})
    with pytest.raises(ValidationError):
        apply_event(Blueprint(project_id="p1"), bad)


def test_a_status_record_without_new_evidence_adds_no_source():
    store = EventStore("p1")
    store.update_field(FieldName.ALGORITHM, "XGBoost", FieldStatus.KNOWN, 0.5, prov("use XGBoost", "M1"), "extractor")
    store.update_field(FieldName.ALGORITHM, "XGBoost", FieldStatus.CONFIRMED, 0.9, None, "state_classifier")
    f = store.state.fields[FieldName.ALGORITHM]
    assert f.status == FieldStatus.CONFIRMED and f.confidence == 0.9
    assert len(f.provenance) == 1
    assert store.rebuild().model_dump(mode="json") == store.state.model_dump(mode="json")