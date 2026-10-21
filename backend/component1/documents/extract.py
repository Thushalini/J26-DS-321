"""Find planning facts in document chunks: rules on every chunk, the language model only on a few.

    facts = extract_document(chunks, fallback_llm=llm, max_llm_chunks=10)

The rules run on every SENTENCE of every chunk (instant, nothing sent anywhere). Sentences, not
whole chunks, because the chat rules keep one value per field per message and mark everything
unsure if one word sounds unsure; a document paragraph names several options in different
sentences. A chunk the rules found nothing in, but which talks about planning, may be sent to
the language model, up to max_llm_chunks per document, so a long document cannot make hundreds
of slow calls. The same checks as for chat apply: the quoted words must really be in the chunk.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from ..extraction.extractor import LLMFn, extract_items
from ..extraction.rules import looks_like_planning, rule_items
from .chunk import Chunk, split_sentences

log = logging.getLogger("component1")

DEFAULT_MAX_LLM_CHUNKS = 10


@dataclass(frozen=True)
class DocFact:
    chunk_no: int
    page: Optional[int]
    field: str
    value: str
    span: str          # the exact words in the chunk
    hedged: bool
    method: str        # "rules" or "llm"


def max_llm_chunks_from_env() -> int:
    raw = (os.environ.get("C1_DOC_MAX_LLM_CHUNKS") or "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_MAX_LLM_CHUNKS
    except ValueError:
        log.warning("C1_DOC_MAX_LLM_CHUNKS=%r is not a whole number. Using %d.", raw, DEFAULT_MAX_LLM_CHUNKS)
        return DEFAULT_MAX_LLM_CHUNKS


def extract_document(
    chunks: Iterable[Chunk],
    fallback_llm: Optional[LLMFn] = None,
    max_llm_chunks: Optional[int] = None,
) -> list[DocFact]:
    if max_llm_chunks is None:
        max_llm_chunks = max_llm_chunks_from_env()
    facts: list[DocFact] = []
    llm_used = 0
    for chunk in chunks:
        items = [item for sentence in split_sentences(chunk.text) for item in rule_items(sentence)]
        method = "rules"
        if not items and fallback_llm is not None and llm_used < max_llm_chunks and looks_like_planning(chunk.text):
            llm_used += 1
            method = "llm"
            try:
                items = extract_items(chunk.text, fallback_llm)
            except Exception as exc:  # slow, rate-limited or offline: this chunk just yields nothing
                log.warning("Document chunk %d: language model failed (%s)", chunk.number, type(exc).__name__)
                items = []
        for item in items:
            facts.append(DocFact(chunk.number, chunk.page, item.field.value, item.value.strip(),
                                 item.span, item.hedged, method))
    return facts
