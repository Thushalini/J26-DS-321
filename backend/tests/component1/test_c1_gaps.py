"""Tests for component1/gaps/gaps.py. Values are invented fixtures."""
from component1.common.events import EventStore
from component1.common.schema import FieldName, FieldStatus, Provenance, SourceType
from component1.gaps.gaps import MIN_CONFIDENCE, check_completeness, ready_gaps


def put(store, name, status, confidence=0.9, value="something"):
    prov = Provenance(source_type=SourceType.CHAT, source_id="M1", speaker="Member_A", text_span="x")
    store.update_field(name, value, status, confidence, prov, "test")


def fill_all(store, status=FieldStatus.CONFIRMED, confidence=0.9):
    for name in FieldName:
        put(store, name, status, confidence)


# ------------------------------------------------------------ gap detection
def test_on_an_empty_blueprint_only_the_problem_is_ready_to_ask():
    assert ready_gaps(EventStore("p1").state) == [FieldName.PROBLEM]


def test_once_the_problem_is_known_its_dependants_unlock_by_weight():
    store = EventStore("p1")
    put(store, FieldName.PROBLEM, FieldStatus.KNOWN)
    assert ready_gaps(store.state) == [FieldName.OBJECTIVE, FieldName.DATASET, FieldName.DOMAIN]


def test_an_uncertain_field_does_not_unlock_the_next_ones():
    store = EventStore("p1")
    put(store, FieldName.PROBLEM, FieldStatus.UNCERTAIN)
    assert ready_gaps(store.state) == []


def test_a_conflicting_field_does_not_unlock_the_next_ones():
    store = EventStore("p1")
    put(store, FieldName.PROBLEM, FieldStatus.CONFLICTING)
    assert ready_gaps(store.state) == []


def test_no_gaps_when_every_field_has_a_value():
    store = EventStore("p1")
    fill_all(store)
    assert ready_gaps(store.state) == []


# --------------------------------------------------------------- completeness
def test_an_empty_blueprint_is_not_complete_and_lists_every_field_as_missing():
    report = check_completeness(EventStore("p1").state)
    assert report.complete is False
    assert set(report.missing) == set(FieldName)


def test_all_fields_confirmed_with_good_confidence_is_complete():
    store = EventStore("p1")
    fill_all(store)
    report = check_completeness(store.state)
    assert report.complete is True
    assert not (report.missing or report.conflicting or report.not_confirmed or report.low_confidence)


def test_one_conflicting_field_blocks_completeness_and_is_named():
    store = EventStore("p1")
    fill_all(store)
    put(store, FieldName.TARGET_VARIABLE, FieldStatus.CONFLICTING)
    report = check_completeness(store.state)
    assert report.complete is False
    assert report.conflicting == [FieldName.TARGET_VARIABLE]


def test_known_or_uncertain_fields_are_reported_as_not_confirmed():
    store = EventStore("p1")
    fill_all(store)
    put(store, FieldName.ALGORITHM, FieldStatus.KNOWN)
    put(store, FieldName.EVALUATION_METRIC, FieldStatus.UNCERTAIN)
    report = check_completeness(store.state)
    assert set(report.not_confirmed) == {FieldName.ALGORITHM, FieldName.EVALUATION_METRIC}
    assert report.complete is False


def test_a_confirmed_field_below_the_minimum_confidence_is_reported():
    store = EventStore("p1")
    fill_all(store)
    put(store, FieldName.DATASET, FieldStatus.CONFIRMED, confidence=MIN_CONFIDENCE - 0.1)
    report = check_completeness(store.state)
    assert report.low_confidence == [FieldName.DATASET]
    assert report.complete is False