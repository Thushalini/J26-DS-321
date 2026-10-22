"""Conflicts by meaning: known names, abbreviations and the NLI rule (with a pretend NLI model)."""
from component1.common.events import EventStore
from component1.common.schema import FieldName, FieldStatus, Provenance, SourceType
from component1.conflicts import nli
from component1.conflicts.same import same_value, statement
from component1.decisions.tracker import find_active, propose
from component1.state.features import field_features, mentions_for_field

CONTRA = {"contradiction": 0.95, "entailment": 0.02, "neutral": 0.03}
IMPLIES = {"contradiction": 0.02, "entailment": 0.90, "neutral": 0.08}
NEUTRAL = {"contradiction": 0.05, "entailment": 0.10, "neutral": 0.85}


def pretend(table):
    """table maps (value_a, value_b) to the result for statement(a) -> statement(b)."""
    def relation(premise, hypothesis):
        for (a, b), out in table.items():
            if premise.endswith(f"is {a}.") and hypothesis.endswith(f"is {b}."):
                return out
        return NEUTRAL
    return relation


def test_abbreviations_are_the_same_choice():
    assert same_value("algorithm", "RF", "random forest")
    assert same_value("algorithm", "GBM", "gradient boosting")
    assert same_value("algorithm", "ANN", "neural network")
    assert same_value("evaluation_metric", "f-measure", "F1")


def test_different_known_names_differ_without_nli():
    assert not same_value("algorithm", "random forest", "SVM")


def test_free_text_needs_nli_when_spelling_differs():
    assert same_value("target_variable", "Tourist Arrivals", "tourist arrivals")
    assert not same_value("target_variable", "monthly tourist arrivals", "tourist arrivals")   # NLI off


def test_nli_contradiction_is_a_conflict():
    nli.set_relation(pretend({("tourist arrivals", "hotel occupancy"): CONTRA}))
    assert not same_value("target_variable", "tourist arrivals", "hotel occupancy")


def test_nli_implication_without_contradiction_is_the_same():
    nli.set_relation(pretend({("monthly tourist arrivals", "tourist arrivals"): IMPLIES}))
    assert same_value("target_variable", "monthly tourist arrivals", "tourist arrivals")


def test_contradiction_in_either_direction_wins():
    nli.set_relation(pretend({("a b", "a"): IMPLIES, ("a", "a b"): CONTRA}))
    assert not same_value("target_variable", "a b", "a")


def test_neutral_both_ways_stays_different():
    nli.set_relation(pretend({}))
    assert not same_value("dataset", "kaggle hotels", "census 2020")


def test_nli_is_not_used_for_named_fields():
    nli.set_relation(lambda p, h: IMPLIES)
    assert not same_value("algorithm", "random forest", "SVM")
    assert not same_value("timeline", "4 weeks", "6 weeks")


def test_statement_template():
    assert statement("target_variable", "churn") == "The target variable of the project is churn."


def _mention(store, field, value, speaker):
    store.update_field(field, value, FieldStatus.KNOWN, 0.8,
                       Provenance(source_type=SourceType.CHAT, source_id="m", speaker=speaker, text_span=value), "extractor")


def test_mentions_group_same_meaning():
    store = EventStore("p1")
    _mention(store, FieldName.ALGORITHM, "random forest", "M-1")
    _mention(store, FieldName.ALGORITHM, "RF", "M-2")
    keys = {m["value"] for m in mentions_for_field(store.events, "algorithm")}
    assert len(keys) == 1
    assert field_features(store.events, "algorithm")[1] == 1.0     # full agreement


def test_find_active_matches_by_meaning():
    store = EventStore("p1")
    propose(store, "algorithm", "random forest", "M-1", "m1", "")
    assert find_active(store.state, "algorithm", "RF") is not None
