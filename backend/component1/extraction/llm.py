"""Pick the language model from backend/.env: Groq or Gemini.

    C1_LLM_PROVIDER=groq      use Groq    (GROQ_API_KEY, GROQ_MODEL)
    C1_LLM_PROVIDER=gemini    use Gemini  (GEMINI_API_KEY, GEMINI_MODEL)
    C1_LLM_PROVIDER=          auto: Groq if GROQ_API_KEY is set, else Gemini if GEMINI_API_KEY is set

Both return the same kind of function (prompt in, reply text out), so nothing else changes
when you switch. Comparing the two on the same labelled messages is one setting away.
"""
from __future__ import annotations

import os
from typing import Optional

from .extractor import LLMFn
from .gemini_provider import GeminiConfigError, make_gemini_llm
from .groq_provider import GroqConfigError, make_groq_llm
from .hybrid import make_hybrid_llm

PROVIDERS = ("groq", "gemini")


class LLMConfigError(RuntimeError):
    """No language model can be used with the current settings."""


def chosen_provider() -> str:
    """The provider the settings ask for. Raises LLMConfigError if none can be used."""
    wanted = (os.environ.get("C1_LLM_PROVIDER") or "").strip().lower()
    if wanted and wanted not in PROVIDERS:
        raise LLMConfigError(f"C1_LLM_PROVIDER must be one of {', '.join(PROVIDERS)}, or left empty.")
    if wanted:
        return wanted
    if (os.environ.get("GROQ_API_KEY") or "").strip():
        return "groq"
    if (os.environ.get("GEMINI_API_KEY") or "").strip():
        return "gemini"
    raise LLMConfigError("No language model key in backend/.env (GROQ_API_KEY or GEMINI_API_KEY).")


def make_configured_llm(provider: Optional[str] = None) -> LLMFn:
    """Build the chosen provider's function. Raises LLMConfigError with a plain reason."""
    provider = provider or chosen_provider()
    try:
        if provider == "groq":
            llm = make_groq_llm()
        elif provider == "gemini":
            llm = make_gemini_llm()
        else:
            raise LLMConfigError(f"Unknown provider {provider!r}.")
    except (GroqConfigError, GeminiConfigError) as exc:
        raise LLMConfigError(f"{provider}: {exc}") from exc
    llm.provider = provider  # for logs and tests
    return llm


def build_extractor_llm(mode: Optional[str] = None) -> tuple[LLMFn, str]:
    """What the app and the demos extract with: rules, plus the configured model as a fallback.

    mode: hybrid / rules / llm, or None for C1_EXTRACTOR_MODE. Returns (function, plain description).
    Never raises because a key is missing: without a model it is rules only, and the description says why.
    """
    try:
        fallback: Optional[LLMFn] = make_configured_llm()
        reason = ""
    except LLMConfigError as exc:
        fallback, reason = None, str(exc)
    llm = make_hybrid_llm(fallback, mode)
    if fallback is None:
        return llm, f"rules only ({reason})"
    name = f"{fallback.provider} ({getattr(fallback, 'model', '?')})"
    if llm.mode == "rules":
        return llm, "rules only (C1_EXTRACTOR_MODE=rules)"
    if llm.mode == "llm":
        return llm, f"{name} for every message"
    return llm, f"rules first, {name} only when the rules find nothing"
