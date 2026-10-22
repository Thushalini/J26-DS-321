"""Tests for component1/state/normalize.py."""
from component1.state.normalize import normalize_value


def test_three_wordings_of_the_same_dataset_count_as_one_value():
    a = normalize_value("Sri Lanka tourism dataset")
    b = normalize_value("sri lankan tourism dataset")
    c = normalize_value("dataset as sri lankan tourism dataset")
    assert a == b == c


def test_a_different_dataset_is_still_a_different_value():
    assert normalize_value("UCI Online Retail") != normalize_value("Sri Lanka tourism dataset")


def test_the_old_behaviour_still_works():
    assert normalize_value("XGBoost") == normalize_value("xgboost ") == normalize_value("The XGBoost.")


def test_a_leading_label_is_removed():
    assert normalize_value("algorithm is XGBoost") == normalize_value("XGBoost")


def test_underscores_hyphens_and_plurals_do_not_create_a_difference():
    assert normalize_value("tourist_arrival") == normalize_value("tourist arrivals")
    assert normalize_value("random-forest") == normalize_value("random forest")


def test_a_really_different_target_is_still_different():
    assert normalize_value("hotel occupancy") != normalize_value("tourist arrivals")


def test_a_double_s_word_is_not_cut():
    assert normalize_value("class") == "class"