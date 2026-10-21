"""Groq as the language model behind the extractor (the same small function as Gemini's).

Groq runs open models (for example openai/gpt-oss-120b) on hardware built for speed.

Setup (in backend/.env, which git ignores):
    GROQ_API_KEY=your-key                from https://console.groq.com/keys
    GROQ_MODEL=<a model name from the Groq console>
    GROQ_TIMEOUT_SECONDS=30              optional. A slow call is given up after this
    GROQ_REASONING_EFFORT=               optional. low / medium / high, only for reasoning models
                                         such as gpt-oss. Empty = the model's default
Install the library once:  pip install groq
"""
from __future__ import annotations

import os
from typing import Optional

from .extractor import LLMFn

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 1                     # one retry, on rate limits and server errors only
REASONING_EFFORTS = ("low", "medium", "high")


class GroqConfigError(RuntimeError):
    """Something needed to call Groq is missing."""


def _env_float(name: str) -> Optional[float]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise GroqConfigError(f"{name} in backend/.env must be a number, or left empty.") from None


def request_options(model: str, reasoning_effort: Optional[str]) -> dict:
    """Everything sent with each call except the prompt."""
    options: dict = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    return options


def make_groq_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    client=None,
    timeout_seconds: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> LLMFn:
    """Return a function that sends a prompt to Groq and returns the reply text.

    client: only for tests. Pass a fake object with chat.completions.create(...).
    """
    model = model or (os.environ.get("GROQ_MODEL") or "").strip()
    if not model:
        raise GroqConfigError("No model name. Set GROQ_MODEL in backend/.env to a model from the Groq console.")
    if timeout_seconds is None:
        timeout_seconds = _env_float("GROQ_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS
    if timeout_seconds <= 0:
        raise GroqConfigError("GROQ_TIMEOUT_SECONDS must be more than 0.")
    if reasoning_effort is None:
        reasoning_effort = (os.environ.get("GROQ_REASONING_EFFORT") or "").strip().lower() or None
    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
        raise GroqConfigError(f"GROQ_REASONING_EFFORT must be one of {', '.join(REASONING_EFFORTS)}, or left empty.")
    options = request_options(model, reasoning_effort)

    if client is None:
        key = api_key or (os.environ.get("GROQ_API_KEY") or "").strip()
        if not key:
            raise GroqConfigError("No API key. Set GROQ_API_KEY in backend/.env.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise GroqConfigError("The groq package is not installed. Run: pip install groq") from exc
        client = Groq(api_key=key, timeout=timeout_seconds, max_retries=MAX_RETRIES)

    def call(prompt: str) -> str:
        response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], **options)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        # A refused or empty reply has no content. Treat it as "nothing found".
        return getattr(choices[0].message, "content", None) or ""

    call.model = model  # for logs and the demos
    return call
