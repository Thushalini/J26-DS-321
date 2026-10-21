"""Split pages into short chunks, each one small enough to read as one piece of evidence.

    chunks = chunk_pages(pages)    # [Chunk(number=1, page=3, text="We will use ..."), ...]

Paragraphs are kept together where possible, and a chunk never crosses a page, so every fact
keeps its page number. Reading stops at the reference list ("References", "Bibliography"),
because cited papers describe OTHER people's datasets and models, not the team's plan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .parse import Page

MAX_CHARS = 1200
MIN_CHARS = 20        # shorter pieces (page numbers, stray headers) are dropped

# A line that is only the heading of the reference list, optionally numbered ("10. References").
REFERENCES_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:references?(?:\s+list)?|bibliography|works\s+cited|literature\s+cited)\s*:?\s*$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# A paragraph that is only a page number or punctuation ("12", "- 3 -", "Page 4 of 9").
_NOISE = re.compile(r"[\d\W_]*|page\s+\d+(?:\s+of\s+\d+)?", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    number: int               # 1, 2, 3 ... across the whole document
    page: Optional[int]
    text: str


def split_sentences(text: str) -> list[str]:
    """The sentences of a chunk, cut after . ! or ? followed by a space."""
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def _clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # words broken across lines: "classi-\nfier"
    return " ".join(text.split())


def _paragraphs(page_text: str) -> tuple[list[str], bool]:
    """(paragraphs, reached_references). A paragraph is text between blank lines."""
    paragraphs, current = [], []
    for line in page_text.splitlines():
        if REFERENCES_HEADING.match(line):
            if current:
                paragraphs.append(_clean("\n".join(current)))
            return [p for p in paragraphs if p], True
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append(_clean("\n".join(current)))
            current = []
    if current:
        paragraphs.append(_clean("\n".join(current)))
    return [p for p in paragraphs if p], False


def _pieces(paragraph: str, max_chars: int) -> list[str]:
    """A paragraph longer than max_chars is cut between sentences (or, as a last resort, words)."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(paragraph):
        while len(sentence) > max_chars:            # one huge "sentence" with no full stop
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_pages(pages: Iterable[Page], max_chars: int = MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []

    def add(page: Optional[int], text: str) -> None:
        if len(text) >= MIN_CHARS:
            chunks.append(Chunk(len(chunks) + 1, page, text))

    for page in pages:
        paragraphs, stop = _paragraphs(page.text)
        current = ""
        for paragraph in paragraphs:
            if _NOISE.fullmatch(paragraph):
                continue
            for piece in _pieces(paragraph, max_chars):
                if current and len(current) + 1 + len(piece) > max_chars:
                    add(page.number, current)
                    current = piece
                else:
                    current = f"{current} {piece}".strip()
        if current:
            add(page.number, current)
        if stop:
            break
    return chunks
