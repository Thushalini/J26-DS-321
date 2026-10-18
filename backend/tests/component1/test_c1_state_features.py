"""Tests for the normaliser, the feature builder, the predictor and the pipeline model.

The events below are small stand-ins with the same shape as real events, so most of
these tests need no database and no Pydantic. All values are invented.
"""
from types import SimpleNamespace

import joblib
from sklearn.ensemble import RandomForestClassifier

from component1.state.features import FEATURE_ORDER, field_features
from component1.state.normalize import normalize_value
from component1.state.predict import load_model, predict_status
from component1.state.simulate import FEATURE_NAMES, generate_dataset
from component1.state.train import SAVED_SET, train_and_evaluate


def ev(version, field, value, speaker, status="known", actor="extractor", event_type="field_updated"):
    return SimpleNamespace(
        version=version,
        event_type=event_type,
        actor=actor,
        payload={"field": field, "new_value": value, "new_status": status,
                 "provenance": {"speaker": speaker}},
    )


# ---------------------------------------------------------------- normalise
def test_case_spaces_and_edge_punctuation_do_not_matter():
    assert normalize_value("XGBoost") == normalize_value("  xgboost. ")


def test_a_leading_article_is_ignored():
    assert normalize_value("The Tourism  Dataset.") == "tourism dataset"
    assert normalize_value("an RMSE") == "rmse"


def test_different_values_stay_different():
    assert normalize_value("tourist arrivals") != normalize_value("hotel occupancy")


def test_an_empty_value_normalises_to_nothing():
    assert normalize_value("  . ") == ""


# ------------------------------------------------------------------ features
def test_a_field_that_was_never_mentioned_has_no_features():
    assert field_features([], "dataset") is None


def test_one_plain_mention():
    assert field_features([ev(1, "algorithm", "XGBoost", "Member_A")], "algorithm") == [1.0, 1.0, 0.0, 1.0]


def test_one_hedged_mention_sets_latest_hedged():
    events = [ev(1, "target_variable", "tourist arrivals", "Member_A", status="uncertain")]
    assert field_features(events, "target_variable") == [1.0, 1.0, 1.0, 1.0]


def test_the_same_person_saying_it_again_is_not_a_second_speaker():
    events = [ev(1, "algorithm", "XGBoost", "Member_A"), ev(2, "algorithm", "XGBoost", "Member_A")]
    assert field_features(events, "algorithm") == [2.0, 1.0, 0.0, 1.0]


def test_another_person_restating_the_same_value_counts_as_a_second_speaker():
    events = [ev(1, "algorithm", "XGBoost", "Member_A"), ev(2, "algorithm", "xgboost ", "Member_B")]
    assert field_features(events, "algorithm") == [2.0, 1.0, 0.0, 2.0]


def test_two_people_with_different_values_lower_the_agreement():
    events = [ev(1, "target_variable", "tourist arrivals", "Member_A"),
              ev(2, "target_variable", "hotel occupancy", "Member_C")]
    assert field_features(events, "target_variable") == [2.0, 0.5, 0.0, 2.0]


def test_latest_hedged_looks_at_the_most_recent_mention_only():
    events = [ev(1, "metric", "RMSE", "Member_A", status="uncertain"), ev(2, "metric", "RMSE", "Member_A")]
    assert field_features(events, "metric")[2] == 0.0


def test_events_are_read_in_version_order_even_if_given_out_of_order():
    events = [ev(2, "metric", "RMSE", "Member_A"), ev(1, "metric", "RMSE", "Member_A", status="uncertain")]
    assert field_features(events, "metric")[2] == 0.0


def test_other_fields_other_actors_and_other_event_types_are_ignored():
    events = [
        ev(1, "dataset", "X", "Member_A"),                                   # another field
        ev(2, "algorithm", "XGBoost", "Member_A", actor="state_classifier"),  # not an extraction
        ev(3, "algorithm", "XGBoost", "Member_A", event_type="scores_updated"),
        ev(4, "algorithm", "XGBoost", "Member_B"),                           # the only real mention
    ]
    assert field_features(events, "algorithm") == [1.0, 1.0, 0.0, 1.0]


def test_an_enum_style_event_type_and_field_name_are_accepted():
    event = ev(1, "algorithm", "XGBoost", "Member_A", event_type=SimpleNamespace(value="field_updated"))
    assert field_features([event], SimpleNamespace(value="algorithm")) == [1.0, 1.0, 0.0, 1.0]


def test_the_feature_order_matches_the_trained_model():
    assert FEATURE_ORDER == [FEATURE_NAMES[i] for i in (0, 1, 4, 5)]


# ----------------------------------------------------------------- predictor
def small_bundle(features=None):
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": features or list(FEATURE_ORDER)}


def test_the_predictor_gives_sensible_labels_for_clear_cases():
    bundle = small_bundle()
    assert predict_status(bundle, [1.0, 1.0, 0.0, 1.0])[0] == "known"
    assert predict_status(bundle, [1.0, 1.0, 1.0, 1.0])[0] == "uncertain"
    assert predict_status(bundle, [2.0, 0.5, 0.0, 2.0])[0] == "conflicting"
    assert predict_status(bundle, [2.0, 1.0, 0.0, 2.0])[0] == "confirmed"


def test_the_probability_is_between_zero_and_one():
    _, probability = predict_status(small_bundle(), [1.0, 1.0, 0.0, 1.0])
    assert 0.0 < probability <= 1.0


def test_a_field_with_no_mentions_is_missing():
    assert predict_status(small_bundle(), None) == ("missing", 1.0)


def test_a_model_trained_on_other_features_is_refused_with_a_clear_message():
    bundle = small_bundle(features=FEATURE_NAMES)
    try:
        predict_status(bundle, [1.0, 1.0, 0.0, 1.0])
    except ValueError as exc:
        assert "Retrain" in str(exc)
        return
    raise AssertionError("expected ValueError")


# ------------------------------------------------------------------ training
def test_training_saves_the_pipeline_model_and_it_loads_back(tmp_path):
    results = train_and_evaluate(
        n=1500, n_estimators=30, save=True,
        model_path=tmp_path / "m.joblib", results_path=tmp_path / "r.json",
    )
    assert (tmp_path / "r.json").exists()
    bundle = load_model(tmp_path / "m.joblib")
    assert bundle["features"] == FEATURE_ORDER
    assert predict_status(bundle, [2.0, 1.0, 0.0, 2.0])[0] == "confirmed"
    sets = results["feature_sets"]
    assert SAVED_SET in sets
    gap = abs(sets[SAVED_SET]["random_forest"]["macro_f1"] - sets["extended_6_features"]["random_forest"]["macro_f1"])
    assert gap < 0.05


# ------------------------------------------- with the real event log (Pydantic)
def test_features_come_out_right_from_the_real_event_store():
    import json

    from component1.common.events import EventStore
    from component1.extraction.extractor import apply_message

    def fake(value, span):
        item = {"field": "algorithm", "value": value, "span": span, "hedged": False}
        return lambda prompt: json.dumps({"items": [item]})

    store = EventStore("p1")
    apply_message(store, "M1", "Member_A", "Let's use XGBoost", fake("XGBoost", "use XGBoost"))
    apply_message(store, "M2", "Member_B", "yes, xgboost it is", fake("xgboost", "xgboost"))
    assert field_features(store.events, "algorithm") == [2.0, 1.0, 0.0, 2.0]