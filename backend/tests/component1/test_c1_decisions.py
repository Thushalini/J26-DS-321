"""Tests for component1/decisions: decision records (tracker), Domino validation and the view.

Uses the keyword intent rules (no trained intent model) and rules-only extraction. All messages
are invented.
"""
import pytest
from sklearn.ensemble import RandomForestClassifier

from component1.agent.chat_agent import ChatAgent
from component1.agent.signals import CHECK
from component1.common.events import EventStore
from component1.common.schema import DecisionStatus as S, FieldName, FieldStatus, Provenance, SourceType
from component1.decisions import tracker as T
from component1.decisions.domino import descendants, restore_passing, run_domino, settle
from component1.decisions.view import decisions_view
from component1.extraction.hybrid import make_hybrid_llm
from component1.pipeline import state_signals
from component1.plan import plan_from_events
from component1.state.features import FEATURE_ORDER
from component1.state.simulate import FEATURE_NAMES, generate_dataset

RULES = make_hybrid_llm(None, mode="rules")


def small_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


BUNDLE = small_bundle()


def status(store, decision_id):
    return store.state.decisions[decision_id].status


def put_field(store, field, value):
    prov = Provenance(source_type=SourceType.CHAT, source_id="m", speaker="M-1", text_span=value)
    store.update_field(FieldName(field), value, FieldStatus.KNOWN, 0.5, prov, "extractor")


# -------------------------------------------------------------- reasons
@pytest.mark.parametrize("text, why", [
    ("Let's use XGBoost because it handles tabular data well.", "it handles tabular data well"),
    ("RMSE, since outliers matter less here", "outliers matter less here"),
    ("We pick F1 so that both classes count", "both classes count"),
    ("Use ARIMA due to the strong seasonality!", "the strong seasonality"),
])
def test_the_reason_is_taken_from_the_message(text, why):
    assert T.reason_in(text) == why


def test_messages_without_a_reason_have_none():
    assert T.reason_in("Let's use XGBoost.") is None


# ---------------------------------------------------------------- life
def test_a_new_value_is_proposed_with_its_reason_and_source():
    store = EventStore("t")
    (c,) = T.on_statement(store, "algorithm", "XGBoost", "M-1", "m1", "Let's use XGBoost because it is fast.")
    d = store.state.decisions[c.decision_id]
    assert (c.kind, d.status, d.proposed_by, d.rationale, d.source_id) == \
        ("proposed", S.PROPOSED, "M-1", "it is fast", "m1")


def test_someone_else_saying_the_same_value_confirms_it_and_the_proposer_repeating_does_not():
    store = EventStore("t")
    (c,) = T.on_statement(store, "algorithm", "XGBoost", "M-1", "m1", "XGBoost")
    assert T.on_statement(store, "algorithm", "xgboost", "M-1", "m2", "xgboost again") == []
    T.on_statement(store, "algorithm", "XGBoost", "M-2", "m3", "Yes XGBoost")
    d = store.state.decisions[c.decision_id]
    assert d.status == S.CONFIRMED and d.agreed_by == ["M-2"]


def test_finalizing_needs_someone_other_than_the_proposer():
    store = EventStore("t")
    (c,) = T.on_statement(store, "algorithm", "XGBoost", "M-1", "m1", "XGBoost")
    d = store.state.decisions[c.decision_id]
    assert T.on_reply(store, d, "M-1", "finalization") == []                     # alone: nothing
    T.on_reply(store, store.state.decisions[c.decision_id], "M-2", "agreement")
    T.on_reply(store, store.state.decisions[c.decision_id], "M-1", "finalization")
    assert status(store, c.decision_id) == S.FINALIZED


def test_another_person_can_finalize_directly():
    store = EventStore("t")
    (c,) = T.on_statement(store, "algorithm", "XGBoost", "M-1", "m1", "XGBoost")
    T.on_reply(store, store.state.decisions[c.decision_id], "M-2", "finalization")
    d = store.state.decisions[c.decision_id]
    assert d.status == S.FINALIZED and d.agreed_by == ["M-2"]


def test_an_unsupported_proposal_is_rejected_but_a_supported_one_only_gets_an_objection():
    store = EventStore("t")
    (a,) = T.on_statement(store, "algorithm", "LSTM", "M-1", "m1", "LSTM")
    T.on_reply(store, store.state.decisions[a.decision_id], "M-2", "rejection")
    assert status(store, a.decision_id) == S.REJECTED
    (b,) = T.on_statement(store, "evaluation_metric", "RMSE", "M-1", "m2", "RMSE")
    T.on_reply(store, store.state.decisions[b.decision_id], "M-2", "agreement")
    T.on_reply(store, store.state.decisions[b.decision_id], "M-3", "rejection")
    d = store.state.decisions[b.decision_id]
    assert d.status == S.CONFIRMED and d.objected_by == ["M-3"]


def test_changing_your_mind_moves_you_from_objected_to_agreed():
    store = EventStore("t")
    (c,) = T.on_statement(store, "evaluation_metric", "RMSE", "M-1", "m1", "RMSE")
    T.on_reply(store, store.state.decisions[c.decision_id], "M-2", "agreement")
    T.on_reply(store, store.state.decisions[c.decision_id], "M-3", "rejection")
    T.on_reply(store, store.state.decisions[c.decision_id], "M-3", "agreement")
    d = store.state.decisions[c.decision_id]
    assert d.agreed_by == ["M-2", "M-3"] and d.objected_by == []


def test_settling_a_value_supersedes_the_other_proposals_for_that_field():
    store = EventStore("t")
    (a,) = T.on_statement(store, "algorithm", "XGBoost", "M-1", "m1", "XGBoost")
    (b,) = T.on_statement(store, "algorithm", "LSTM", "M-2", "m2", "LSTM")
    changes = T.on_statement(store, "algorithm", "LSTM", "M-3", "m3", "LSTM it is")
    assert status(store, b.decision_id) == S.CONFIRMED and status(store, a.decision_id) == S.SUPERSEDED
    assert {c.kind for c in changes} == {"confirmed", "superseded"}


def test_an_accepted_document_suggestion_is_a_confirmed_decision_citing_the_page():
    store = EventStore("t")
    (c,) = T.record_document_decision(store, "dataset", "UCI Retail", "Plan.pdf", 3, "the UCI Retail data", "M-2", "f1")
    d = store.state.decisions[c.decision_id]
    assert d.status == S.CONFIRMED and d.proposed_by == "doc:Plan.pdf" and d.agreed_by == ["M-2"]
    assert d.rationale == 'Found in Plan.pdf p.3: "the UCI Retail data"'


def test_milestones_are_not_planning_decisions():
    store = EventStore("t")
    assert T.on_statement(store, "timeline", "milestone: Demo, 2026-11-20", "M-1", "m1", "demo") == []


def test_decisions_survive_a_restart():
    store = EventStore("t")
    (c,) = T.on_statement(store, "evaluation_metric", "RMSE", "M-1", "m1", "RMSE because it is simple")
    T.on_reply(store, store.state.decisions[c.decision_id], "M-2", "agreement")
    T.on_reply(store, store.state.decisions[c.decision_id], "M-3", "rejection")
    again = EventStore.from_events("t", store.events).state.decisions[c.decision_id]
    assert again == store.state.decisions[c.decision_id]


# -------------------------------------------------------------- domino
def test_descendants_follow_the_plan_order():
    assert descendants("task_type")[:2] == ["algorithm", "evaluation_metric"]
    assert "resources" in descendants("task_type") and "problem" not in descendants("task_type")


def settled(store, field, value, who=("M-1", "M-2")):
    put_field(store, field, value)
    (c,) = T.on_statement(store, field, value, who[0], "m", value)
    T.on_reply(store, store.state.decisions[c.decision_id], who[1], "finalization")
    return c.decision_id


def test_a_change_that_breaks_a_later_decision_sets_it_to_pending_fix_with_the_cause():
    store = EventStore("t")
    settled(store, "task_type", "classification")
    metric = settled(store, "evaluation_metric", "F1-score")
    put_field(store, "task_type", "regression")
    new_task = settled(store, "task_type", "regression", who=("M-2", "M-3"))
    changes = settle(store, [T.Change("x", "task_type", "classification", "superseded", "M-3")])
    d = store.state.decisions[metric]
    assert d.status == S.PENDING_FIX and d.held_status == S.FINALIZED
    assert d.pending_reason.startswith("the task type changed from classification to regression")
    assert "Consider RMSE or MAE" in d.pending_reason and [c.kind for c in changes] == ["pending_fix"]
    assert status(store, new_task) == S.FINALIZED


def test_a_change_that_still_fits_leaves_later_decisions_alone():
    store = EventStore("t")
    settled(store, "task_type", "classification")
    algo = settled(store, "algorithm", "XGBoost")
    put_field(store, "task_type", "regression")
    settled(store, "task_type", "regression", who=("M-2", "M-3"))
    assert run_domino(store, {"task_type": ("classification", "regression")}) == []
    assert status(store, algo) == S.FINALIZED                                      # XGBoost does regression too


def test_unchecked_finalized_decisions_directly_below_are_flagged_for_a_human_re_check():
    store = EventStore("t")
    settled(store, "dataset", "UCI Retail")
    target = settled(store, "target_variable", "churn")
    proposed = T.on_statement(store, "task_type", "classification", "M-1", "m9", "classification")[0].decision_id
    run_domino(store, {"dataset": ("UCI Retail", "Kaggle Telco")})
    assert status(store, target) == S.PENDING_FIX
    assert "please re-check it" in store.state.decisions[target].pending_reason
    assert status(store, proposed) == S.PROPOSED                                    # only settled ones move


def test_a_re_check_clears_when_the_check_passes_again_or_someone_agrees():
    store = EventStore("t")
    settled(store, "task_type", "classification")
    metric = settled(store, "evaluation_metric", "F1-score")
    put_field(store, "task_type", "regression")
    settled(store, "task_type", "regression", who=("M-2", "M-3"))
    run_domino(store, {"task_type": ("classification", "regression")})
    put_field(store, "task_type", "classification")                                 # the team changes back
    settled(store, "task_type", "classification", who=("M-3", "M-1"))
    restore_passing(store)
    assert status(store, metric) == S.FINALIZED and store.state.decisions[metric].pending_reason is None

    store2 = EventStore("t")
    settled(store2, "dataset", "UCI Retail")
    target = settled(store2, "target_variable", "churn")
    run_domino(store2, {"dataset": ("UCI Retail", "Kaggle Telco")})
    T.on_reply(store2, store2.state.decisions[target], "M-3", "agreement")           # a person re-confirms
    assert status(store2, target) == S.CONFIRMED


def test_a_pending_decision_gets_one_message_naming_the_cause_not_two():
    store = EventStore("t")
    settled(store, "task_type", "classification")
    settled(store, "evaluation_metric", "F1-score")
    put_field(store, "task_type", "regression")
    settled(store, "task_type", "regression", who=("M-2", "M-3"))
    run_domino(store, {"task_type": ("classification", "regression")})
    checks = [s for s in state_signals(store) if s.kind == CHECK and s.field == "evaluation_metric"]
    assert len(checks) == 1 and checks[0].detail.startswith("the task type changed")


def test_a_settled_decision_ends_the_field_level_disagreement():
    store = EventStore("t")
    T.on_statement(store, "algorithm", "XGBoost", "M-1", "m1", "XGBoost")
    T.on_statement(store, "algorithm", "LSTM", "M-2", "m2", "LSTM")
    assert T.resolved_by_decision(store.state, "algorithm", "conflicting") == "conflicting"
    T.on_statement(store, "algorithm", "LSTM", "M-3", "m3", "LSTM")
    assert T.resolved_by_decision(store.state, "algorithm", "conflicting") == "confirmed"


# ------------------------------------------------------- through the chat
def test_a_planning_chat_records_decisions_and_flags_the_broken_one():
    agent = ChatAgent("t", BUNDLE, RULES)                       # keyword intents, no trained model
    chat = [("M-1", "This is a classification task."), ("M-2", "Agreed, classification it is."),
            ("M-1", "Let's use F1 score because the classes are imbalanced."), ("M-2", "Sounds good."),
            ("M-2", "Ok, let's lock it in."), ("M-2", "Let's use XGBoost since it handles tabular data well."),
            ("M-3", "No, I don't think XGBoost is a good idea."), ("M-2", "Actually this is a regression task."),
            ("M-1", "Yes, regression.")]
    replies = []
    for i, (who, text) in enumerate(chat):
        replies += agent.on_message(f"m{i}", who, text, now=i * 200)
    view = decisions_view(agent.store.state, agent.store.events)
    by_field = {d["field"]: d for d in view["current"]}
    assert by_field["evaluation_metric"]["status"] == "pending_fix"
    assert by_field["evaluation_metric"]["rationale"] == "the classes are imbalanced"
    assert by_field["task_type"]["value"] == "regression" and by_field["task_type"]["status"] == "confirmed"
    assert "algorithm" not in by_field                              # XGBoost was rejected
    assert any("proposal rejected by M-3" in r for r in replies)
    assert any("decision finalized" in r and "the classes are imbalanced" in r for r in replies)
    assert any("may no longer fit" in r for r in replies)


# ------------------------------------------------------------------ view
def test_the_view_has_one_current_decision_per_field_with_its_timeline():
    store = EventStore("t")
    (c,) = T.on_statement(store, "algorithm", "XGBoost", "M-1", "m1", "XGBoost because it is fast")
    T.on_reply(store, store.state.decisions[c.decision_id], "M-2", "agreement")
    view = decisions_view(store.state, store.events)
    (d,) = view["current"]
    assert (d["value"], d["status_text"], d["agreed_by"]) == ("XGBoost", "confirmed", ["M-2"])
    assert [t["what"] for t in d["timeline"]] == ["proposed", "confirmed"]
    assert view["needs_recheck"] == []


def test_the_view_can_hide_documents_and_reasons():
    store = EventStore("t")
    T.record_document_decision(store, "dataset", "UCI Retail", "Ayesha_notes.docx", None, "UCI Retail", "M-2", "f1")
    (d,) = decisions_view(store.state, store.events, hide_documents=True)["current"]
    assert d["proposed_by"] == "a document" and d["rationale"] is None
    (e,) = decisions_view(store.state, store.events, include_evidence=False)["current"]
    assert e["rationale"] is None


def test_the_plan_carries_the_decisions():
    store = EventStore("t")
    T.on_statement(store, "algorithm", "XGBoost", "M-1", "m1", "XGBoost")
    out = plan_from_events("t", store.events)
    assert [d["value"] for d in out["decisions"]["current"]] == ["XGBoost"]


# ------------------------------------------------ found by manual testing
def test_a_plain_agreed_confirms_the_field_not_just_the_decision():
    agent = ChatAgent("t", BUNDLE, RULES)
    agent.on_message("m1", "M-1", "Let's use F1 score as the metric.", 0)
    agent.on_message("m2", "M-2", "Agreed.", 20)
    assert agent.store.state.fields[FieldName.EVALUATION_METRIC].status == FieldStatus.CONFIRMED


def test_restating_and_finalizing_in_one_message_gives_one_line():
    agent = ChatAgent("t", BUNDLE, RULES)
    agent.on_message("m1", "M-1", "This is a classification task.", 0)
    replies = agent.on_message("m2", "M-2", "Classification, agreed, let's finalize it.", 20)
    text = "\n".join(replies)
    assert text.count("task type = classification") == 1
    assert "decision finalized" in text


def test_a_field_waiting_for_a_recheck_is_no_longer_confirmed():
    agent = ChatAgent("t", BUNDLE, RULES)
    for i, (who, text) in enumerate([
        ("M-1", "This is a classification task."), ("M-2", "Agreed, finalize it."),
        ("M-1", "We will use accuracy as the metric."), ("M-2", "Agreed, finalize it."),
        ("M-3", "Actually it should be a regression task."), ("M-1", "Agreed, finalize it."),
    ]):
        agent.on_message(f"m{i + 1}", who, text, i * 20)
    state = agent.store.state
    assert any(d.status == S.PENDING_FIX and d.field == FieldName.EVALUATION_METRIC for d in state.decisions.values())
    assert state.fields[FieldName.EVALUATION_METRIC].status != FieldStatus.CONFIRMED


def test_naming_a_value_to_turn_it_down_is_an_objection():
    store = EventStore("t")
    T.propose(store, "resources", "AWS", "M-1", "m1")
    T.on_statement(store, "resources", "AWS", "M-2", "m2", "No, AWS costs too much.", intent="rejection")
    d = next(iter(store.state.decisions.values()))
    assert d.status == S.REJECTED and d.objected_by == ["M-2"]


def test_a_new_value_after_a_settled_decision_is_a_conflict_not_confirmed():
    store = EventStore("t")
    put_field(store, "algorithm", "Random Forest")
    T.propose(store, "algorithm", "Random Forest", "M-1", "m1", status=S.CONFIRMED, agreed_by=["M-2"])
    put_field(store, "algorithm", "Neural Network")
    assert T.resolved_by_decision(store.state, "algorithm", "conflicting") == "conflicting"
    assert T.resolved_by_decision(store.state, "algorithm", "confirmed") == "confirmed"


def test_a_rejected_value_is_not_a_rival_any_more():
    store = EventStore("t")
    put_field(store, "resources", "AWS")
    T.propose(store, "resources", "AWS", "M-1", "m1")
    T.object_to(store, next(iter(store.state.decisions.values())), "M-2")
    put_field(store, "resources", "CPU only")
    assert T.resolved_by_decision(store.state, "resources", "conflicting") == "known"
    assert T.turned_down(store.state, "resources", "AWS")


def test_a_rejected_value_on_the_field_is_not_confirmed():
    store = EventStore("t")
    put_field(store, "resources", "AWS")
    T.propose(store, "resources", "AWS", "M-1", "m1")
    T.object_to(store, next(iter(store.state.decisions.values())), "M-2")
    assert T.resolved_by_decision(store.state, "resources", "confirmed") == "uncertain"
