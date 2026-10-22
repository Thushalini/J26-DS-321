"""Tests for component1/extraction/rules.py and hybrid.py.

The messages are invented to exercise the rules. They are not data.
"""
import json

import pytest

from component1.common.events import EventStore
from component1.common.schema import FieldName, FieldStatus
from component1.extraction.extractor import apply_message, build_prompt, extract_items, message_from_prompt
from component1.extraction.hybrid import make_hybrid_llm, mode_from_env
from component1.extraction.rules import looks_like_planning, rule_items


def found(message):
    return {i.field.value: i.value for i in rule_items(message)}


def item(message, field):
    return next(i for i in rule_items(message) if i.field.value == field)


# ----------------------------------------------------------- word lists
@pytest.mark.parametrize("message, field, value", [
    ("Let's use XGBoost, it handles tabular data well.", "algorithm", "XGBoost"),
    ("random forests could work", "algorithm", "Random Forest"),
    ("We could forecast monthly arrivals using ARIMA", "algorithm", "ARIMA"),
    ("RMSE is what we report", "evaluation_metric", "RMSE"),
    ("let's use F1 score", "evaluation_metric", "F1-score"),
    ("this is a classification task", "task_type", "classification"),
    ("We can train on Google Colab", "resources", "Google Colab"),
])
def test_known_names_are_found_and_written_the_standard_way(message, field, value):
    assert found(message)[field] == value


def test_logistic_regression_is_an_algorithm_not_the_task_regression():
    got = found("We should use logistic regression")
    assert got["algorithm"] == "Logistic Regression"
    assert "task_type" not in got


def test_names_inside_other_words_are_not_matched():
    assert "algorithm" not in found("the gruesome deadline")      # not GRU
    assert "evaluation_metric" not in found("maestro")            # not MAE


# ------------------------------------------------------------ patterns
@pytest.mark.parametrize("message, value", [
    ("I think we should predict tourist arrivals", "tourist arrivals"),
    ("Actually the target should be hotel occupancy.", "hotel occupancy"),
    ("shall we have our target group as tourist_arrival", "tourist_arrival"),
    ("predict churn using XGBoost", "churn"),
])
def test_the_target_variable_is_read_from_phrases(message, value):
    assert found(message)["target_variable"] == value


def test_vague_words_are_not_saved_as_a_target():
    assert "target_variable" not in found("Can we predict it?")


@pytest.mark.parametrize("message, value", [
    ("We can use the Kaggle tourism dataset", "Kaggle tourism dataset"),
    ("the dataset is UCI Online Retail", "UCI Online Retail"),
])
def test_named_datasets_are_found(message, value):
    assert found(message)["dataset"] == value


@pytest.mark.parametrize("message", ["We need more training data", "There is a lot of data in the folder"])
def test_talk_about_data_is_not_a_dataset_name(message):
    assert "dataset" not in found(message)


def test_the_domain_is_only_taken_when_stated_not_guessed():
    assert found("so our group is tourism")["domain"] == "tourism"
    assert found("We'll do it in the healthcare domain")["domain"] == "healthcare"
    assert "domain" not in found("I think we should predict tourist arrivals")


def test_problem_and_objective_need_an_explicit_phrase():
    assert found("Our goal is to reduce hotel overbooking")["objective"] == "reduce hotel overbooking"
    assert "hotels can't plan staff" in found("The problem is that hotels can't plan staff for peak season")["problem"]


# --------------------------------------------- hedging, rejection, several
def test_unsure_wording_and_questions_are_marked_hedged():
    assert item("Maybe RMSE for the metric?", "evaluation_metric").hedged
    assert item("Should we use XGBoost", "algorithm").hedged
    assert not item("We will use XGBoost.", "algorithm").hedged


def test_a_rejected_option_is_not_saved():
    got = found("Don't use accuracy, the classes are imbalanced. Let's use F1.")
    assert got["evaluation_metric"] == "F1-score"
    assert "algorithm" not in found("we should not use XGBoost here")


def test_two_values_for_one_field_keep_the_first_and_mark_it_hedged():
    i = item("Let's compare random forest and XGBoost", "algorithm")
    assert i.value == "Random Forest" and i.hedged


def test_the_span_is_copied_exactly_from_the_message():
    message = "We could forecast monthly arrivals using ARIMA"
    for i in rule_items(message):
        assert i.span in message


@pytest.mark.parametrize("message", ["hello team", "What time is the meeting tomorrow?", "I'll push my code tonight.", "", "   "])
def test_small_talk_gives_nothing(message):
    assert rule_items(message) == []


def test_planning_cues():
    assert looks_like_planning("which model should we pick")
    assert not looks_like_planning("see you tomorrow")


# ------------------------------------------------------------- hybrid
class CountingLLM:
    def __init__(self, reply='{"items": []}', fail=False):
        self.reply, self.fail, self.calls = reply, fail, 0

    def __call__(self, prompt):
        self.calls += 1
        if self.fail:
            raise TimeoutError("too slow")
        return self.reply


def test_the_prompt_round_trips_to_the_message():
    message = "Let's use XGBoost.\nAnd RMSE."
    assert message_from_prompt(build_prompt(message)) == message


def test_hybrid_does_not_call_the_llm_when_the_rules_found_something():
    llm = CountingLLM()
    items = extract_items("Let's use XGBoost.", make_hybrid_llm(llm, mode="hybrid"))
    assert [i.value for i in items] == ["XGBoost"] and llm.calls == 0


def test_hybrid_does_not_call_the_llm_for_small_talk():
    llm = CountingLLM()
    assert extract_items("see you tomorrow", make_hybrid_llm(llm, mode="hybrid")) == []
    assert llm.calls == 0


def test_hybrid_asks_the_llm_when_planning_talk_has_no_rule_match():
    reply = json.dumps({"items": [{"field": "objective", "value": "help hotels plan", "span": "help hotels plan"}]})
    llm = CountingLLM(reply)
    items = extract_items("our model should help hotels plan", make_hybrid_llm(llm, mode="hybrid"))
    assert llm.calls == 1 and items[0].field == FieldName.OBJECTIVE


def test_a_failing_llm_fallback_gives_nothing_instead_of_raising():
    llm = CountingLLM(fail=True)
    assert extract_items("our model should help hotels plan", make_hybrid_llm(llm, mode="hybrid")) == []
    assert llm.calls == 1


def test_rules_mode_never_calls_the_llm():
    llm = CountingLLM()
    extract_items("our model should help hotels plan", make_hybrid_llm(llm, mode="rules"))
    assert llm.calls == 0


def test_llm_mode_always_calls_the_llm():
    llm = CountingLLM()
    extract_items("Let's use XGBoost.", make_hybrid_llm(llm, mode="llm"))
    assert llm.calls == 1


def test_without_an_llm_every_mode_becomes_rules_only():
    for mode in ("hybrid", "llm"):
        fn = make_hybrid_llm(None, mode=mode)
        assert fn.mode == "rules"
        assert extract_items("Let's use XGBoost.", fn)[0].value == "XGBoost"


def test_an_unknown_mode_is_refused_but_a_bad_env_value_falls_back(monkeypatch):
    with pytest.raises(ValueError):
        make_hybrid_llm(None, mode="magic")
    monkeypatch.setenv("C1_EXTRACTOR_MODE", "magic")
    assert mode_from_env() == "hybrid"
    monkeypatch.setenv("C1_EXTRACTOR_MODE", " Rules ")
    assert mode_from_env() == "rules"


def test_rule_output_goes_into_the_blueprint_like_llm_output():
    store = EventStore("p1")
    apply_message(store, "M1", "Member_A", "Maybe RMSE for the metric?", make_hybrid_llm(None, mode="rules"))
    f = store.state.fields[FieldName.EVALUATION_METRIC]
    assert f.value == "RMSE" and f.status == FieldStatus.UNCERTAIN
