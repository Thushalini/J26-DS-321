"""Keyword baseline: typical words, code and file names -> lifecycle stage."""
import pytest

from component2.classify.keyword_baseline import classify


@pytest.mark.parametrize("text, stage", [
    ("Clean missing values with dropna", "data_cleaning"),
    ("Plot histogram of the age distribution", "eda"),
    ("Encode categories with get_dummies and StandardScaler", "feature_engineering"),
    ("Train XGBoost with GridSearchCV hyperparameter tuning", "model_development"),
    ("Print confusion_matrix and f1_score on the test set", "model_evaluation"),
    ("Add Dockerfile and deploy the FastAPI endpoint", "deployment_mlops"),
    ("Update README with setup notes", "documentation"),
])
def test_typical_items_get_their_stage(text, stage):
    assert classify(text)[0] == stage


def test_no_clue_is_other_with_zero_confidence():
    assert classify("ok see you tomorrow") == ("other", 0.0)


def test_confidence_is_the_top_stages_share_of_hits():
    stage, confidence = classify("Clean missing values with dropna")
    assert stage == "data_cleaning" and confidence == 1.0
    _, mixed = classify("Clean missing values, then plot a histogram")      # 2 cleaning hits + 2 eda hits
    assert mixed == 0.5


def test_short_markdown_cell_is_documentation():
    assert classify("# Results", cell_type="markdown") == ("documentation", 0.6)
    assert classify("x" * 250 + " train the model", cell_type="markdown")[0] == "model_development"


def test_whole_words_only_but_code_clues_match_inside_code():
    assert classify("modelling notes")[0] != "model_development"           # 'model' is not a whole word here
    assert classify("clf.fit(X, y)")[0] == "model_development"            # '.fit(' matches as written
