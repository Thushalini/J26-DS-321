"""Tests for component1/extraction/groq_provider.py and llm.py. No internet and no API key are used."""
import json
from types import SimpleNamespace

import pytest

from component1.extraction.extractor import extract_items
from component1.extraction.groq_provider import GroqConfigError, make_groq_llm, request_options
from component1.extraction.llm import LLMConfigError, chosen_provider, make_configured_llm


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.content is None:
            return SimpleNamespace(choices=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class FakeGroq:
    def __init__(self, content='{"items": []}'):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))

    @property
    def calls(self):
        return self.chat.completions.calls


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("GROQ_API_KEY", "GROQ_MODEL", "GROQ_TIMEOUT_SECONDS", "GROQ_REASONING_EFFORT",
                 "GEMINI_API_KEY", "GEMINI_MODEL", "C1_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# --------------------------------------------------------------- provider
def test_the_reply_text_is_returned(clean_env):
    assert make_groq_llm(model="m", client=FakeGroq('{"items": [1]}'))("hi") == '{"items": [1]}'


def test_it_sends_the_prompt_as_one_user_message_with_json_output(clean_env):
    client = FakeGroq()
    make_groq_llm(model="some-model", client=client)("my prompt")
    call = client.calls[0]
    assert call["model"] == "some-model"
    assert call["messages"] == [{"role": "user", "content": "my prompt"}]
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in call


def test_reasoning_effort_is_sent_only_when_set(clean_env):
    client = FakeGroq()
    clean_env.setenv("GROQ_REASONING_EFFORT", " Low ")
    make_groq_llm(model="m", client=client)("x")
    assert client.calls[0]["reasoning_effort"] == "low"


def test_a_bad_reasoning_effort_or_timeout_gives_a_clear_error(clean_env):
    with pytest.raises(GroqConfigError, match="GROQ_REASONING_EFFORT"):
        make_groq_llm(model="m", client=FakeGroq(), reasoning_effort="extreme")
    clean_env.setenv("GROQ_TIMEOUT_SECONDS", "soon")
    with pytest.raises(GroqConfigError, match="GROQ_TIMEOUT_SECONDS"):
        make_groq_llm(model="m", client=FakeGroq())
    with pytest.raises(GroqConfigError):
        make_groq_llm(model="m", client=FakeGroq(), timeout_seconds=0)


def test_an_empty_reply_becomes_an_empty_string(clean_env):
    assert make_groq_llm(model="m", client=FakeGroq(None))("x") == ""
    assert make_groq_llm(model="m", client=FakeGroq(""))("x") == ""


def test_missing_model_or_key_give_clear_errors(clean_env):
    with pytest.raises(GroqConfigError, match="GROQ_MODEL"):
        make_groq_llm(client=FakeGroq())
    with pytest.raises(GroqConfigError, match="GROQ_API_KEY"):
        make_groq_llm(model="m")


def test_the_model_can_come_from_the_environment(clean_env):
    clean_env.setenv("GROQ_MODEL", "env-model")
    client = FakeGroq()
    make_groq_llm(client=client)("x")
    assert client.calls[0]["model"] == "env-model"


def test_request_options_shape():
    assert request_options("m", None) == {"model": "m", "temperature": 0, "response_format": {"type": "json_object"}}


def test_the_real_client_can_be_built_offline(clean_env):
    """Builds (does not call) a real client, so the option names match the installed library."""
    pytest.importorskip("groq")
    make_groq_llm(model="m", api_key="test-key-not-real")


def test_groq_plugs_into_the_extractor(clean_env):
    reply = json.dumps({"items": [{"field": "algorithm", "value": "XGBoost", "span": "use XGBoost", "hedged": False}]})
    items = extract_items("Let's use XGBoost.", make_groq_llm(model="m", client=FakeGroq(reply)))
    assert items[0].value == "XGBoost"


# ---------------------------------------------------------- choosing one
def test_auto_prefers_groq_then_gemini(clean_env):
    with pytest.raises(LLMConfigError):
        chosen_provider()
    clean_env.setenv("GEMINI_API_KEY", "g")
    assert chosen_provider() == "gemini"
    clean_env.setenv("GROQ_API_KEY", "q")
    assert chosen_provider() == "groq"


def test_an_explicit_provider_wins(clean_env):
    clean_env.setenv("GROQ_API_KEY", "q")
    clean_env.setenv("C1_LLM_PROVIDER", "Gemini")
    assert chosen_provider() == "gemini"
    clean_env.setenv("C1_LLM_PROVIDER", "openai")
    with pytest.raises(LLMConfigError, match="C1_LLM_PROVIDER"):
        chosen_provider()


def test_missing_settings_become_one_plain_error(clean_env):
    clean_env.setenv("C1_LLM_PROVIDER", "groq")
    with pytest.raises(LLMConfigError, match="groq: .*GROQ_MODEL"):
        make_configured_llm()


def test_a_configured_provider_is_labelled(clean_env):
    pytest.importorskip("groq")
    clean_env.setenv("GROQ_API_KEY", "test-key-not-real")
    clean_env.setenv("GROQ_MODEL", "m")
    assert make_configured_llm().provider == "groq"


# ------------------------------------------------- what the app and demos use
def test_build_extractor_llm_without_a_key_is_rules_only_and_says_why(clean_env):
    from component1.extraction.llm import build_extractor_llm

    llm, how = build_extractor_llm()
    assert llm.mode == "rules" and how.startswith("rules only (")
    assert extract_items("Let's use XGBoost.", llm)[0].value == "XGBoost"


def test_build_extractor_llm_describes_the_hybrid_with_the_provider_and_model(clean_env):
    pytest.importorskip("groq")
    from component1.extraction.llm import build_extractor_llm

    clean_env.setenv("GROQ_API_KEY", "test-key-not-real")
    clean_env.setenv("GROQ_MODEL", "some-model")
    llm, how = build_extractor_llm("hybrid")
    assert llm.mode == "hybrid" and how == "rules first, groq (some-model) only when the rules find nothing"
    assert build_extractor_llm("rules")[1] == "rules only (C1_EXTRACTOR_MODE=rules)"
    assert build_extractor_llm("llm")[1] == "groq (some-model) for every message"
