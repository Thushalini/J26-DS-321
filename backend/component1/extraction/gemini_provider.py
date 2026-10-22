"""Gemini as the language model behind the extractor (one small function).

The extractor only needs a function: prompt text in, reply text out.
This file builds that function for Gemini. Nothing else in the project depends
on Gemini, so using another provider later means writing another small file.

Setup (in backend/.env, which git ignores):
    GEMINI_API_KEY=your-key
    GEMINI_MODEL=<a current Gemini model name from Google AI Studio>
    GEMINI_TIMEOUT_SECONDS=30     optional. A slow call is given up after this, so it cannot block the chat
    GEMINI_THINKING_BUDGET=       optional. Leave empty for the model's default. 0 turns thinking off,
                                  but only on models that allow it (flash-lite rejects it with a 400 error)
Install the library once:  pip install google-genai
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

LLMFn = Callable[[str], str]

DEFAULT_TIMEOUT_SECONDS = 30.0
# Total tries per call, so 2 = one retry. The library retries only on rate limits and server errors.
MAX_ATTEMPTS = 2


class GeminiConfigError(RuntimeError):
    """Something needed to call Gemini is missing."""


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file into the environment.

    Variables that are already set win. Values are never printed.
    """
    path = Path(path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_number(name: str, cast):
    """Read an optional number from the environment. Empty or unset means None."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return cast(raw)
    except ValueError:
        raise GeminiConfigError(f"{name} in backend/.env must be a number, or left empty.") from None


def http_options(timeout_seconds: float) -> dict:
    """Time limit and retry limit for the Gemini client. The library wants the timeout in milliseconds."""
    return {
        "timeout": int(timeout_seconds * 1000),
        "retry_options": {"attempts": MAX_ATTEMPTS},
    }


def generation_config(thinking_budget: Optional[int]) -> dict:
    """The settings sent with every call. Thinking is only mentioned when a budget is set."""
    config: dict = {"temperature": 0, "response_mime_type": "application/json"}
    if thinking_budget is not None:
        config["thinking_config"] = {"thinking_budget": thinking_budget}
    return config


def make_gemini_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    client=None,
    timeout_seconds: Optional[float] = None,
    thinking_budget: Optional[int] = None,
) -> LLMFn:
    """Return a function that sends a prompt to Gemini and returns the reply text.

    timeout_seconds and thinking_budget fall back to GEMINI_TIMEOUT_SECONDS and
    GEMINI_THINKING_BUDGET. A call that times out raises, and the caller skips that message.
    client: only for tests. Pass a fake object with models.generate_content(...).
    """
    model = model or os.environ.get("GEMINI_MODEL")
    if not model:
        raise GeminiConfigError(
            "No model name. Set GEMINI_MODEL in backend/.env to a current Gemini "
            "model name from Google AI Studio."
        )
    if timeout_seconds is None:
        timeout_seconds = _env_number("GEMINI_TIMEOUT_SECONDS", float) or DEFAULT_TIMEOUT_SECONDS
    if timeout_seconds <= 0:
        raise GeminiConfigError("GEMINI_TIMEOUT_SECONDS must be more than 0.")
    if thinking_budget is None:
        thinking_budget = _env_number("GEMINI_THINKING_BUDGET", int)
    config = generation_config(thinking_budget)

    if client is None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise GeminiConfigError("No API key. Set GEMINI_API_KEY in backend/.env.")
        try:
            from google import genai
        except ImportError as exc:
            raise GeminiConfigError(
                "The google-genai package is not installed. Run: pip install google-genai"
            ) from exc
        client = genai.Client(api_key=key, http_options=http_options(timeout_seconds))

    def call(prompt: str) -> str:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        # A blocked or empty reply can come back with no text. Treat it as "nothing found".
        return getattr(response, "text", None) or ""

    call.model = model  # for logs and the demos
    return call