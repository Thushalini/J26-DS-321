"""Gemini as the language model behind the extractor (one small function).

The extractor only needs a function: prompt text in, reply text out.
This file builds that function for Gemini. Nothing else in the project depends
on Gemini, so using another provider later means writing another small file.

Setup (in backend/.env, which git ignores):
    GEMINI_API_KEY=your-key
    GEMINI_MODEL=<a current Gemini model name from Google AI Studio>
Install the library once:  pip install google-genai
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

LLMFn = Callable[[str], str]


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


def make_gemini_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    client=None,
) -> LLMFn:
    """Return a function that sends a prompt to Gemini and returns the reply text.

    client: only for tests. Pass a fake object with models.generate_content(...).
    """
    model = model or os.environ.get("GEMINI_MODEL")
    if not model:
        raise GeminiConfigError(
            "No model name. Set GEMINI_MODEL in backend/.env to a current Gemini "
            "model name from Google AI Studio."
        )
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
        client = genai.Client(api_key=key)

    def call(prompt: str) -> str:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0, "response_mime_type": "application/json"},
        )
        # A blocked or empty reply can come back with no text. Treat it as "nothing found".
        return getattr(response, "text", None) or ""

    return call