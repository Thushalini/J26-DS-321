"""Tests for component1/agent/policy.py. Invented values, plain times in seconds."""
from component1.agent.policy import Buddy, BuddyConfig
from component1.agent.signals import CHECK, CONFLICT, FINALIZED, FOUND, GAP, SAVED, Signal

CFG = BuddyConfig()
TARGET_CONFLICT = Signal(CONFLICT, "target_variable", ("tourist arrivals", "hotel occupancy"))
METRIC_CHECK = Signal(
    CHECK, "evaluation_metric",
    detail="RMSE measures numeric error, but the task type is classification.",
    source="rule table",
)


# ---------------------------------------------------------------- silence
def test_no_signals_means_silence():
    assert Buddy().step([], now=0) == []


# --------------------------------------------------------------- conflicts
def test_a_conflict_is_raised_straight_away_and_names_both_values():
    buddy = Buddy()
    buddy.heard_message(0)
    (text,) = buddy.step([TARGET_CONFLICT], now=0)
    assert "target variable" in text
    assert "tourist arrivals" in text and "hotel occupancy" in text


def test_the_same_conflict_is_not_repeated_soon():
    buddy = Buddy()
    buddy.step([TARGET_CONFLICT], now=0)
    assert buddy.step([TARGET_CONFLICT], now=60) == []


def test_the_same_conflict_is_raised_again_after_the_repeat_window():
    buddy = Buddy()
    buddy.step([TARGET_CONFLICT], now=0)
    assert len(buddy.step([TARGET_CONFLICT], now=CFG.repeat_after_seconds + 1)) == 1


def test_a_conflict_with_new_values_counts_as_a_new_question():
    buddy = Buddy()
    buddy.step([TARGET_CONFLICT], now=0)
    changed = Signal(CONFLICT, "target_variable", ("tourist arrivals", "hotel occupancy", "revenue"))
    (text,) = buddy.step([changed], now=30)
    assert "revenue" in text


def test_three_values_are_joined_in_plain_english():
    text = Buddy().step([Signal(CONFLICT, "algorithm", ("a", "b", "c"))], now=0)[0]
    assert "a, b and c" in text


# -------------------------------------------------------------------- gaps
def test_a_gap_is_not_asked_while_people_are_still_chatting():
    buddy = Buddy()
    buddy.heard_message(100)
    assert buddy.step([Signal(GAP, "dataset")], now=105) == []


def test_a_gap_is_asked_once_the_chat_has_been_quiet():
    buddy = Buddy()
    buddy.heard_message(100)
    (text,) = buddy.step([Signal(GAP, "dataset")], now=100 + CFG.quiet_seconds_before_gap_question)
    assert "dataset" in text


def test_a_gap_is_asked_right_after_a_decision_is_finalized():
    buddy = Buddy()
    buddy.heard_message(100)
    (text,) = buddy.step([Signal(FINALIZED, "algorithm"), Signal(GAP, "evaluation_metric")], now=101)
    assert "evaluation metric" in text


def test_gap_questions_are_rate_limited_but_conflicts_are_not():
    buddy = Buddy()
    buddy.step([Signal(GAP, "dataset")], now=0)                               # Buddy speaks
    assert buddy.step([Signal(GAP, "objective")], now=10) == []               # too soon
    assert len(buddy.step([TARGET_CONFLICT], now=10)) == 1                    # conflicts still allowed
    assert len(buddy.step([Signal(GAP, "objective")], now=10 + CFG.min_seconds_between_questions)) == 1


def test_only_one_message_per_step_and_conflicts_come_first():
    buddy = Buddy()
    replies = buddy.step([Signal(GAP, "dataset"), TARGET_CONFLICT], now=500)
    assert len(replies) == 1 and "different answers" in replies[0]


def test_only_the_top_gap_is_asked():
    buddy = Buddy()
    (text,) = buddy.step([Signal(GAP, "problem"), Signal(GAP, "dataset")], now=500)
    assert "problem" in text and "dataset" not in text


def test_the_same_gap_is_not_asked_twice_but_the_next_one_is():
    buddy = Buddy()
    gaps = [Signal(GAP, "problem"), Signal(GAP, "dataset")]
    buddy.step(gaps, now=0)
    (text,) = buddy.step(gaps, now=CFG.min_seconds_between_questions + 1)
    assert "dataset" in text


def test_a_gap_can_be_asked_again_after_the_repeat_window():
    buddy = Buddy()
    buddy.step([Signal(GAP, "problem")], now=0)
    assert len(buddy.step([Signal(GAP, "problem")], now=CFG.repeat_after_seconds + 1)) == 1


# --------------------------------------------------- announcing saved fields
def test_a_saved_field_is_announced_with_its_value_and_speaker():
    buddy = Buddy()
    (text,) = buddy.step([Signal(SAVED, "algorithm", ("XGBoost",), source="Member_A")], now=0)
    assert text == "Noted: algorithm = XGBoost (from Member_A)."


def test_a_saved_uncertain_field_says_so():
    buddy = Buddy()
    saved = Signal(SAVED, "target_variable", ("tourist arrivals",), detail="marked uncertain", source="Member_A")
    (text,) = buddy.step([saved], now=0)
    assert "marked uncertain" in text and "tourist arrivals" in text


def test_the_same_save_is_not_announced_twice():
    buddy = Buddy()
    saved = Signal(SAVED, "algorithm", ("XGBoost",), source="Member_A")
    buddy.step([saved], now=0)
    assert buddy.step([saved], now=10) == []


def test_a_changed_status_is_announced_again():
    buddy = Buddy()
    buddy.step([Signal(SAVED, "metric", ("RMSE",), detail="marked uncertain")], now=0)
    assert len(buddy.step([Signal(SAVED, "metric", ("RMSE",), detail="now known")], now=10)) == 1


def test_several_saves_are_combined_into_one_message():
    buddy = Buddy()
    replies = buddy.step(
        [Signal(SAVED, "algorithm", ("XGBoost",)), Signal(SAVED, "evaluation_metric", ("RMSE",))], now=0
    )
    assert len(replies) == 1
    assert "algorithm = XGBoost" in replies[0] and "evaluation metric = RMSE" in replies[0]


def test_an_announcement_does_not_use_up_the_question_rate_limit():
    buddy = Buddy()
    buddy.step([Signal(SAVED, "algorithm", ("XGBoost",))], now=0)
    (text,) = buddy.step([Signal(FINALIZED, "algorithm"), Signal(GAP, "dataset")], now=1)
    assert "dataset" in text


def test_an_announcement_and_a_conflict_arrive_in_one_message_announcement_first():
    buddy = Buddy()
    saved = Signal(SAVED, "target_variable", ("hotel occupancy",), source="Member_C")
    replies = buddy.step([saved, TARGET_CONFLICT], now=0)
    assert len(replies) == 1
    first, second = replies[0].split("\n")
    assert first.startswith("Noted:") and "different answers" in second


# ------------------------------------------------------ checks that find a problem
def test_a_problem_found_by_a_check_is_raised_straight_away_with_its_source():
    buddy = Buddy()
    buddy.heard_message(0)
    (text,) = buddy.step([METRIC_CHECK], now=0)
    assert "evaluation metric" in text and "RMSE" in text and "(source: rule table)" in text


def test_the_same_check_problem_is_not_repeated_but_a_different_one_is():
    buddy = Buddy()
    buddy.step([METRIC_CHECK], now=0)
    assert buddy.step([METRIC_CHECK], now=10) == []
    other = Signal(CHECK, "evaluation_metric", detail="A different problem.", source="rule table")
    assert len(buddy.step([other], now=20)) == 1


def test_a_conflict_comes_before_a_check_problem():
    buddy = Buddy()
    (text,) = buddy.step([METRIC_CHECK, TARGET_CONFLICT], now=0)
    assert "different answers" in text


# ----------------------------------- candidates found in the uploaded documents
def test_a_document_candidate_is_offered_at_a_calm_moment_with_its_source():
    buddy = Buddy()
    (text,) = buddy.step([Signal(FOUND, "dataset", ("Tourism arrivals dataset",), source="Proposal.pdf p.4")], now=0)
    assert "Proposal.pdf p.4" in text and "Tourism arrivals dataset" in text


def test_a_document_candidate_replaces_the_plain_gap_question_for_that_field():
    buddy = Buddy()
    signals = [Signal(GAP, "dataset"), Signal(FOUND, "dataset", ("Tourism arrivals dataset",), source="Proposal.pdf p.4")]
    (text,) = buddy.step(signals, now=0)
    assert text.startswith("I found a possible dataset")


def test_a_document_candidate_waits_while_people_are_chatting():
    buddy = Buddy()
    buddy.heard_message(100)
    assert buddy.step([Signal(FOUND, "dataset", ("X",), source="Proposal.pdf")], now=105) == []