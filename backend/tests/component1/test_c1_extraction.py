"""Tests for component1/extraction/extractor.py. A fake model stands in for the LLM."""
import json

from component1.common.events import EventStore
from component1.common.schema import FieldName, FieldStatus
from component1.extraction.extractor import apply_message, build_prompt, extract_items

MSG = "I think we should predict tourist arrivals."
ITEM = {
    "field": "target_variable",
    "value": "tourist arrivals",
    "span": "predict tourist arrivals",
    "hedged": True,
}


def fake(items):
    return lambda prompt: json.dumps({"items": items})


def test_a_hedged_statement_fills_the_field_as_uncertain():
    store = EventStore("p1")
    events = apply_message(store, "M0003", "Member_A", MSG, fake([ITEM]))
    assert len(events) == 1
    f = store.state.fields[FieldName.TARGET_VARIABLE]
    assert f.value == "tourist arrivals"
    assert f.status == FieldStatus.UNCERTAIN
    assert f.provenance[0].text_span == "predict tourist arrivals"
    assert f.provenance[0].speaker == "Member_A"
    assert f.provenance[0].source_id == "M0003"


def test_a_plain_statement_fills_the_field_as_known():
    store = EventStore("p1")
    item = {"field": "algorithm", "value": "XGBoost", "span": "use XGBoost", "hedged": False}
    apply_message(store, "M0001", "Member_B", "Let's use XGBoost", fake([item]))
    assert store.state.fields[FieldName.ALGORITHM].status == FieldStatus.KNOWN


def test_a_span_that_is_not_in_the_message_is_dropped():
    invented = {"field": "algorithm", "value": "deep learning", "span": "use deep learning", "hedged": False}
    assert extract_items(MSG, fake([invented])) == []


def test_an_unknown_field_name_is_dropped():
    odd = {"field": "favourite_colour", "value": "blue", "span": "tourist arrivals", "hedged": False}
    assert extract_items(MSG, fake([odd])) == []


def test_a_reply_that_is_not_json_gives_nothing_and_does_not_crash():
    assert extract_items(MSG, lambda prompt: "Sorry, I cannot help with that.") == []
    assert extract_items(MSG, lambda prompt: '{"items": "oops"}') == []


def test_json_wrapped_in_a_code_fence_is_still_read():
    reply = "```json\n" + json.dumps({"items": [ITEM]}) + "\n```"
    assert len(extract_items(MSG, lambda prompt: reply)) == 1


def test_the_prompt_contains_the_message():
    seen = []

    def spy(prompt):
        seen.append(prompt)
        return '{"items": []}'

    extract_items(MSG, spy)
    assert MSG in seen[0] and MSG in build_prompt(MSG)


def test_nothing_extracted_means_no_events():
    store = EventStore("p1")
    assert apply_message(store, "M0002", "Member_C", "What time is the meeting?", fake([])) == []
    assert store.state.version == 0


def test_a_changed_value_keeps_the_old_one_in_the_history():
    store = EventStore("p1")
    first = {"field": "target_variable", "value": "tourist arrivals", "span": "tourist arrivals", "hedged": False}
    second = {"field": "target_variable", "value": "hotel occupancy", "span": "hotel occupancy", "hedged": False}
    apply_message(store, "M0003", "Member_A", "predict tourist arrivals", fake([first]))
    apply_message(store, "M0010", "Member_C", "the target is hotel occupancy", fake([second]))
    assert store.state.fields[FieldName.TARGET_VARIABLE].value == "hotel occupancy"
    assert store.state.version == 2
    assert store.events[1].payload["old_value"] == "tourist arrivals"