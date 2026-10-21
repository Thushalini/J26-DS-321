"""Rules first, the language model only when the rules found nothing.

    llm = make_hybrid_llm(gemini_llm)     # or make_hybrid_llm(None) for rules only

The result has the same shape as any LLM function (prompt in, JSON text out), so it plugs
into the existing extractor, pipeline and service without changing them. The extractor's
checks (known field names, the quoted words really in the message) apply to rule output too.

Modes (C1_EXTRACTOR_MODE in backend/.env):
    hybrid  default. Rules; the LLM only for planning-like messages the rules could not read.
    rules   never call the LLM. Instant, free, offline, nothing leaves this machine.
    llm     always call the LLM (the original behaviour), for comparing the two in the report.
A failed or timed-out LLM call is logged and treated as "nothing found", so it never breaks the chat.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from .extractor import LLMFn, message_from_prompt
from .rules import looks_like_planning, rule_items

log = logging.getLogger("component1")

MODES = ("hybrid", "rules", "llm")
EMPTY = '{"items": []}'


def mode_from_env() -> str:
    mode = (os.environ.get("C1_EXTRACTOR_MODE") or "hybrid").strip().lower()
    if mode not in MODES:
        log.warning("C1_EXTRACTOR_MODE=%r is not one of %s. Using 'hybrid'.", mode, ", ".join(MODES))
        return "hybrid"
    return mode


def make_hybrid_llm(fallback: Optional[LLMFn], mode: Optional[str] = None) -> LLMFn:
    """Return a prompt -> JSON function that tries the rules before the fallback LLM.

    fallback: the real LLM function, or None to use rules only whatever the mode says.
    """
    mode = mode or mode_from_env()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if fallback is None and mode != "rules":
        log.info("No language model configured: Component 1 extracts with rules only.")
        mode = "rules"

    def call(prompt: str) -> str:
        if mode == "llm":
            return fallback(prompt)
        message = message_from_prompt(prompt)
        items = rule_items(message)
        if items or mode == "rules" or not looks_like_planning(message):
            return json.dumps({"items": [i.model_dump(mode="json") for i in items]})
        try:
            return fallback(prompt)
        except Exception as exc:  # slow, rate-limited or offline: the message just yields nothing
            log.warning("LLM fallback failed (%s). The message is skipped.", type(exc).__name__)
            return EMPTY

    call.mode = mode  # for logs and tests
    # The model behind the rules, for callers that run the rules themselves and decide when to
    # ask the model (documents/extract.py caps it per document). None in "rules" mode.
    call.fallback = fallback if mode != "rules" else None
    return call
