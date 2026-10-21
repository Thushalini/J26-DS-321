"""Tests for component1/history.py: the readable decision history built from the event log.

Events are made by the real pipeline with a scripted language model and a small model trained
on simulated data. All values are invented.
"""
import json

from sklearn.ensemble import RandomForestClassifier

from component1.common.events import EventStore
from component1.common.schema import FieldName, Provenance, SourceType, FieldStatus
from component1.documents.candidates import Candidate
from component1.documents.integrate import accept_candidate
from component1.history import field_history, recent_changes
from component1.pipeline import process_message
from component1.plan import plan_from_events
from component1.state.features import FEATURE_ORDER
from component1.state.simulate import FEATURE_NAMES, generate_dataset


def small_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


BUNDLE = small_bundle()
SCRIPT = {
    "Let's use XGBoost.": ("algorithm", "XGBoost", "use XGBoost", False),
    "I'd rather use LSTM.": ("algorithm", "LSTM", "use LSTM", False),
    "XGBoost again, please.": ("algorithm", "xgboost", "XGBoost", False),
    "Maybe RMSE?": ("evaluation_metric", "RMSE", "RMSE", True),
}


def scripted(prompt):
    for text, (field, value, span, hedged) in SCRIPT.items():
        if text in prompt:
            return json.dumps({"items": [{"field": field, "value": value, "span": span, "hedged": hedged}]})
    return json.dumps({"items": []})


def say(store, msg_id, speaker, text):
    process_message(store, BUNDLE, scripted, msg_id, speaker, text)


def test_a_mention_and_its_judgement_become_one_entry():
    store = EventStore("t")
    say(store, "m1", "M-1", "Let's use XGBoost.")
    (e,) = field_history(store.events)
    assert len(store.events) == 2                       # extractor + state classifier
    assert e["kind"] == "proposed" and e["value"] == "XGBoost" and e["by"] == "M-1"
    assert e["status_before"] == "missing" and e["status_after"] != "missing"
    assert e["summary"] == "M-1 proposed XGBoost" and e["confidence_band"] in ("High", "Medium", "Low")
    assert e["source"] == {"type": "chat", "id": "m1", "page": None}
    assert "text" not in e


def test_changes_agreement_and_repeats_are_told_apart():
    store = EventStore("t")
    say(store, "m1", "M-1", "Let's use XGBoost.")
    say(store, "m2", "M-2", "I'd rather use LSTM.")
    say(store, "m3", "M-1", "XGBoost again, please.")
    say(store, "m4", "M-1", "XGBoost again, please.")
    kinds = [(e["kind"], e["by"], e["previous_value"], e["value"]) for e in field_history(store.events)]
    assert kinds == [
        ("proposed", "M-1", None, "XGBoost"),
        ("changed", "M-2", "XGBoost", "LSTM"),
        ("changed", "M-1", "LSTM", "xgboost"),
        ("repeated", "M-1", "xgboost", "xgboost"),
    ]
    assert field_history(store.events)[1]["summary"] == "M-2 changed it from XGBoost to LSTM"


def test_the_same_value_from_someone_else_is_agreement_and_spelling_does_not_matter():
    store = EventStore("t")
    say(store, "m1", "M-1", "Let's use XGBoost.")
    say(store, "m2", "M-2", "XGBoost again, please.")          # "xgboost" from another person
    second = field_history(store.events)[1]
    assert second["kind"] == "agreed" and second["summary"] == "M-2 agreed on xgboost"


def test_the_status_path_shows_a_conflict_appearing():
    store = EventStore("t")
    say(store, "m1", "M-1", "Let's use XGBoost.")
    say(store, "m2", "M-2", "I'd rather use LSTM.")
    statuses = [(e["status_before"], e["status_after"]) for e in field_history(store.events)]
    assert statuses[1] == (statuses[0][1], "conflicting")


def test_one_field_can_be_asked_for_and_fields_stay_apart():
    store = EventStore("t")
    say(store, "m1", "M-1", "Let's use XGBoost.")
    say(store, "m2", "M-2", "Maybe RMSE?")
    assert [e["field"] for e in field_history(store.events)] == ["algorithm", "evaluation_metric"]
    (metric,) = field_history(store.events, field="evaluation_metric")
    assert metric["label"] == "Evaluation metric" and metric["status_after"] == "uncertain"


def test_quotes_appear_only_when_evidence_is_allowed():
    store = EventStore("t")
    say(store, "m1", "M-1", "Let's use XGBoost.")
    assert field_history(store.events, include_evidence=True)[0]["text"] == "use XGBoost"


def test_an_accepted_document_suggestion_is_one_entry_naming_the_file_and_the_member():
    store = EventStore("t")
    c = Candidate(field="dataset", value="Kaggle tourism dataset", doc_id="d1", filename="Plan.pdf", page=3,
                  span="the Kaggle tourism dataset", mentions=1, fact_ids=["f1"])
    accept_candidate(store, BUNDLE, c, "M-2")
    (e,) = field_history(store.events, include_evidence=True)
    assert e["kind"] == "accepted_document" and e["accepted_by"] == "M-2"
    assert e["document"] == "Plan.pdf" and e["source_label"] == "Plan.pdf p.3" and e["source"]["page"] == 3
    assert e["summary"] == "M-2 accepted Kaggle tourism dataset from Plan.pdf p.3"
    assert e["text"] == "the Kaggle tourism dataset"


def test_file_names_can_be_hidden():
    store = EventStore("t")
    c = Candidate(field="dataset", value="UCI Retail", doc_id="d1", filename="Ayesha_notes.docx", page=None,
                  span="UCI Retail", mentions=1, fact_ids=["f1"])
    accept_candidate(store, BUNDLE, c, "M-2")
    (e,) = field_history(store.events, hide_documents=True)
    assert e["document"] is None and "Ayesha" not in e["summary"] and e["summary"].endswith("from a document")


def test_a_judgement_without_a_mention_is_a_status_change_and_no_change_is_skipped():
    store = EventStore("t")
    say(store, "m1", "M-1", "Let's use XGBoost.")
    field = store.state.fields[FieldName.ALGORITHM]
    store.update_field(FieldName.ALGORITHM, field.value, field.status, 0.9, None, "state_classifier")   # same status
    store.update_field(FieldName.ALGORITHM, field.value, FieldStatus.CONFIRMED, 0.9, None, "state_classifier")
    entries = field_history(store.events)
    assert [e["kind"] for e in entries] == ["proposed", "status_changed"]
    assert entries[-1]["status_after"] == "confirmed" and entries[-1]["summary"] == "Status changed to confirmed"


def test_a_mention_with_no_judgement_after_it_still_shows():
    store = EventStore("t")
    store.update_field(FieldName.DOMAIN, "tourism", FieldStatus.KNOWN, 0.5,
                       Provenance(source_type=SourceType.CHAT, source_id="m1", speaker="M-1", text_span="tourism"),
                       "extractor")
    (e,) = field_history(store.events)
    assert e["kind"] == "proposed" and e["status_after"] == "known"


def test_recent_changes_are_newest_first_and_limited_and_in_the_plan():
    store = EventStore("t")
    for i, text in enumerate(["Let's use XGBoost.", "I'd rather use LSTM.", "Maybe RMSE?"]):
        say(store, f"m{i}", f"M-{i}", text)
    recent = recent_changes(store.events, limit=2)
    assert [e["field"] for e in recent] == ["evaluation_metric", "algorithm"]
    plan = plan_from_events("t", store.events)
    assert [e["version"] for e in plan["recent_changes"]] == sorted(
        (e["version"] for e in plan["recent_changes"]), reverse=True)


def test_an_empty_log_has_no_history():
    assert field_history([]) == [] and plan_from_events("t", [])["recent_changes"] == []
