"""Tests for component1/service.py (the one function the chat endpoint calls).

A pretend language model and a small model trained on simulated data are used, so no
internet, no key and no saved model file are needed. All values are invented.
"""
import json
import os
import time
from contextlib import contextmanager

from sklearn.ensemble import RandomForestClassifier

from component1.service import AgentService, speaker_for, to_seconds
from component1.state.features import FEATURE_ORDER
from component1.state.simulate import FEATURE_NAMES, generate_dataset


def small_bundle():
    X, y = generate_dataset(1500, seed=3)
    cols = [FEATURE_NAMES.index(n) for n in FEATURE_ORDER]
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0).fit(X[:, cols], y)
    return {"model": model, "labels": list(model.classes_), "features": list(FEATURE_ORDER)}


BUNDLE = small_bundle()

TOURIST = "I think we should predict tourist arrivals."
XGB = "Let's use XGBoost, it handles tabular data well."
SCRIPT = {
    TOURIST: [{"field": "target_variable", "value": "tourist arrivals", "span": "predict tourist arrivals", "hedged": True}],
    XGB: [{"field": "algorithm", "value": "XGBoost", "span": "use XGBoost", "hedged": False}],
}
T0 = "2026-10-19T10:00:00Z"


def scripted(prompt):
    for text, items in SCRIPT.items():
        if text in prompt:
            return json.dumps({"items": items})
    return json.dumps({"items": []})


def service(llm=scripted):
    return AgentService(bundle=BUNDLE, llm=llm, env_file=None)


@contextmanager
def env(**changes):
    old = {k: os.environ.get(k) for k in changes}
    try:
        for k, v in changes.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_a_member_message_gets_buddys_reply():
    (reply,) = service().handle_chat_message("t1", "m1", "M-1", TOURIST, T0)
    assert reply.startswith("Noted:") and "tourist arrivals" in reply


def test_buddys_own_messages_are_ignored_and_leave_no_trace():
    svc = service()
    assert svc.handle_chat_message("t1", "m1", None, TOURIST, T0, sender_type="assistant") == []
    assert len(svc.handle_chat_message("t1", "m1", "M-1", TOURIST, T0)) == 1   # the id was not used up


def test_the_same_message_is_processed_only_once():
    svc = service()
    assert len(svc.handle_chat_message("t1", "m1", "M-1", TOURIST, T0)) == 1
    assert svc.handle_chat_message("t1", "m1", "M-1", TOURIST, T0) == []


def test_blank_text_is_ignored():
    assert service().handle_chat_message("t1", "m1", "M-1", "   ", T0) == []


def test_a_failing_language_model_never_raises_and_the_next_message_still_works():
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the model is down")
        return scripted(prompt)

    svc = service(flaky)
    assert svc.handle_chat_message("t1", "m1", "M-1", TOURIST, T0) == []        # no exception
    assert len(svc.handle_chat_message("t1", "m2", "M-1", TOURIST, T0)) == 1   # recovered


def test_a_service_that_is_not_set_up_returns_nothing_instead_of_raising(tmp_path):
    svc = AgentService(bundle=None, llm=None, env_file=None, model_path=tmp_path / "missing.joblib")
    assert svc.handle_chat_message("t1", "m1", "M-1", TOURIST, T0) == []


def test_without_a_gemini_key_the_rules_still_extract():
    svc = AgentService(bundle=BUNDLE, llm=None, env_file=None)
    with env(GEMINI_API_KEY=None, GEMINI_MODEL=None, GROQ_API_KEY=None, GROQ_MODEL=None,
             C1_LLM_PROVIDER=None, C1_EXTRACTOR_MODE=None):
        replies = svc.handle_chat_message("t1", "m1", "M-1", "Let's use XGBoost for this.", T0)
    assert len(replies) == 1 and "algorithm = XGBoost" in replies[0]


def test_the_speaker_falls_back_from_member_code_to_user_id_to_unknown():
    assert speaker_for("M-1", "u9") == "M-1"
    assert speaker_for("  ", "u9") == "user:u9"
    assert speaker_for(None, None) == "unknown"


def test_a_message_without_a_member_code_is_still_handled():
    assert len(service().handle_chat_message("t1", "m1", None, TOURIST, T0, user_id="u9")) == 1


def test_two_teams_are_independent():
    svc = service()
    assert len(svc.handle_chat_message("t1", "m1", "M-1", TOURIST, T0)) == 1
    assert len(svc.handle_chat_message("t2", "m1", "M-2", TOURIST, T0)) == 1   # same message id, other team


def test_the_apps_text_timestamps_are_converted_to_seconds():
    assert to_seconds("2026-10-19T10:03:20Z") - to_seconds(T0) == 200
    assert to_seconds(12.5) == 12.5
    assert abs(to_seconds(None) - time.time()) < 5


def test_a_bad_timestamp_does_not_lose_the_message():
    assert len(service().handle_chat_message("t1", "m1", "M-1", TOURIST, "garbage")) == 1


def test_buddy_asks_about_a_missing_field_on_a_timer_after_a_quiet_spell():
    svc = service()
    svc.handle_chat_message("t1", "m1", "M-1", XGB, T0)
    assert svc.tick("t1", "2026-10-19T10:00:10Z") == []                 # still chatting
    (reply,) = svc.tick("t1", "2026-10-19T10:03:20Z")                   # 200 seconds of quiet
    assert "problem" in reply


def test_ticking_a_team_that_has_no_agent_is_harmless():
    assert service().tick("nobody", T0) == []


def test_tick_all_returns_only_teams_that_have_something_to_say():
    svc = service()
    svc.handle_chat_message("t1", "m1", "M-1", XGB, T0)
    result = svc.tick_all("2026-10-19T10:03:20Z")
    assert list(result) == ["t1"] and len(result["t1"]) == 1