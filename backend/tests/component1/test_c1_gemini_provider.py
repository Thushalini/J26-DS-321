"""Tests for component1/extraction/gemini_provider.py. No internet and no API key are used."""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from component1.extraction.gemini_provider import GeminiConfigError, load_env_file, make_gemini_llm


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.text)


class FakeClient:
    def __init__(self, text='{"items": []}'):
        self.models = FakeModels(text)


@contextmanager
def env(**changes):
    """Set or remove environment variables for one test, then restore them."""
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


def test_the_function_returns_the_reply_text():
    llm = make_gemini_llm(model="m", client=FakeClient('{"items": [1]}'))
    assert llm("hello") == '{"items": [1]}'


def test_it_sends_the_prompt_the_model_temperature_zero_and_json_output():
    client = FakeClient()
    make_gemini_llm(model="some-model", client=client)("my prompt")
    call = client.models.calls[0]
    assert call["model"] == "some-model"
    assert call["contents"] == "my prompt"
    assert call["config"]["temperature"] == 0
    assert call["config"]["response_mime_type"] == "application/json"


def test_an_empty_or_missing_reply_becomes_an_empty_string():
    assert make_gemini_llm(model="m", client=FakeClient(None))("x") == ""
    assert make_gemini_llm(model="m", client=FakeClient(""))("x") == ""


def test_the_model_name_can_come_from_the_environment():
    client = FakeClient()
    with env(GEMINI_MODEL="env-model"):
        make_gemini_llm(client=client)("x")
    assert client.models.calls[0]["model"] == "env-model"


def test_a_missing_model_name_gives_a_clear_error():
    with env(GEMINI_MODEL=None):
        try:
            make_gemini_llm(client=FakeClient())
        except GeminiConfigError as exc:
            assert "GEMINI_MODEL" in str(exc)
            return
    raise AssertionError("expected GeminiConfigError")


def test_a_missing_key_gives_a_clear_error_and_never_shows_one():
    with env(GEMINI_API_KEY=None):
        try:
            make_gemini_llm(model="m")
        except GeminiConfigError as exc:
            assert "GEMINI_API_KEY" in str(exc)
            return
    raise AssertionError("expected GeminiConfigError")


# ------------------------------------------------------------ the .env reader
def test_env_file_lines_are_read_with_quotes_comments_and_export():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / ".env"
        path.write_text(
            "\ufeff# a comment\n\nC1_TEST_A=plain\nC1_TEST_B=\"quoted\"\nexport C1_TEST_C='single'\nnot a pair\n",
            encoding="utf-8",
        )
        with env(C1_TEST_A=None, C1_TEST_B=None, C1_TEST_C=None):
            load_env_file(path)
            assert os.environ["C1_TEST_A"] == "plain"
            assert os.environ["C1_TEST_B"] == "quoted"
            assert os.environ["C1_TEST_C"] == "single"


def test_variables_that_are_already_set_are_not_overwritten():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / ".env"
        path.write_text("C1_TEST_D=from_file\n", encoding="utf-8")
        with env(C1_TEST_D="already_set"):
            load_env_file(path)
            assert os.environ["C1_TEST_D"] == "already_set"


def test_a_missing_env_file_does_nothing():
    load_env_file(Path("this/does/not/exist/.env"))


# ----------------------------------------------- works with the real extractor
def test_the_function_plugs_into_the_extractor():
    import json

    from component1.extraction.extractor import extract_items

    reply = json.dumps({"items": [{
        "field": "target_variable", "value": "tourist arrivals",
        "span": "predict tourist arrivals", "hedged": True,
    }]})
    llm = make_gemini_llm(model="m", client=FakeClient(reply))
    items = extract_items("I think we should predict tourist arrivals.", llm)
    assert len(items) == 1 and items[0].value == "tourist arrivals"

# ------------------------------------------------- time limit and thinking
def test_no_thinking_setting_is_sent_by_default():
    client = FakeClient()
    with env(GEMINI_THINKING_BUDGET=None):
        make_gemini_llm(model="m", client=client)("x")
    assert "thinking_config" not in client.models.calls[0]["config"]


def test_thinking_budget_can_be_turned_off():
    client = FakeClient()
    make_gemini_llm(model="m", client=client, thinking_budget=0)("x")
    assert client.models.calls[0]["config"]["thinking_config"] == {"thinking_budget": 0}


def test_thinking_budget_can_come_from_the_environment():
    client = FakeClient()
    with env(GEMINI_THINKING_BUDGET="0"):
        make_gemini_llm(model="m", client=client)("x")
    assert client.models.calls[0]["config"]["thinking_config"] == {"thinking_budget": 0}


def test_an_empty_thinking_budget_in_the_environment_means_the_default():
    client = FakeClient()
    with env(GEMINI_THINKING_BUDGET=""):
        make_gemini_llm(model="m", client=client)("x")
    assert "thinking_config" not in client.models.calls[0]["config"]


def test_a_bad_number_in_the_environment_gives_a_clear_error():
    for name, bad in (("GEMINI_THINKING_BUDGET", "lots"), ("GEMINI_TIMEOUT_SECONDS", "soon")):
        with env(**{name: bad}):
            try:
                make_gemini_llm(model="m", client=FakeClient())
            except GeminiConfigError as exc:
                assert name in str(exc)
                continue
        raise AssertionError(f"expected GeminiConfigError for {name}")


def test_a_timeout_of_zero_or_less_is_refused():
    try:
        make_gemini_llm(model="m", client=FakeClient(), timeout_seconds=0)
    except GeminiConfigError:
        return
    raise AssertionError("expected GeminiConfigError")


def test_http_options_set_the_time_limit_in_milliseconds_and_one_retry():
    from component1.extraction.gemini_provider import MAX_ATTEMPTS, http_options

    opts = http_options(30)
    assert opts["timeout"] == 30_000
    assert opts["retry_options"]["attempts"] == MAX_ATTEMPTS == 2


def test_the_real_client_accepts_the_http_options():
    """Builds (does not call) a real client, so the option names match the installed library."""
    import pytest

    genai = pytest.importorskip("google.genai")
    from component1.extraction.gemini_provider import http_options

    genai.Client(api_key="test-key-not-real", http_options=http_options(30))
