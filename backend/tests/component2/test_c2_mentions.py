"""Mentions of the three blueprint decisions (target, algorithm, metric) in commits, cells and chat."""
from component2.drift.mentions import count_mentions, find_mentions, normalise_target


def test_many_spellings_of_one_algorithm_become_one_name():
    for text in ("used random forest", "RandomForestClassifier(n_estimators=100)", "RF beats the baseline"):
        assert find_mentions(text)["algorithm"] == {"random forest": 1}


def test_algorithms_and_metrics_found_together():
    found = find_mentions("XGBClassifier gives f1_score 0.81 and accuracy 0.9")
    assert set(found["algorithm"]) == {"xgboost"}
    assert set(found["metric"]) == {"f1-score", "accuracy"}


def test_map_metric_is_case_sensitive():
    assert "map" not in find_mentions("df['x'] = df['x'].map(clean)")["metric"]
    assert "map" in find_mentions("mAP50 went up to 0.62")["metric"]


def test_mean_average_precision_is_not_counted_as_precision():
    assert "precision" not in find_mentions("mean average precision is 0.6")["metric"]


def test_target_column_from_code_and_text():
    assert find_mentions("y = df['Churn']")["target"] == {"churn": 1}
    assert find_mentions("target = 'Sale Price'")["target"] == {"saleprice": 1}
    assert find_mentions("our target variable is Survived")["target"] == {"survived": 1}
    assert not find_mentions("we hit our target this week")["target"]


def test_target_names_compare_equal_after_normalising():
    assert normalise_target(" Sale Price ") == normalise_target("sale_price") == normalise_target("SalePrice")


def test_count_mentions_adds_up_over_texts():
    seen = count_mentions(["random forest", "RF again", "switched to xgboost"])
    assert seen["algorithm"] == {"random forest": 2, "xgboost": 1}
    assert not seen["metric"]
