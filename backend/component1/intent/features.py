"""Turn a message and the message before it into numbers for the classifier."""
from __future__ import annotations

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Turn


class ContextFeatures:
    """Word and character patterns of the message, plus (optionally) word patterns of the
    previous message. use_context=False gives the "message alone" version for the ablation."""

    def __init__(self, use_context: bool = True):
        self.use_context = use_context
        self.message_words = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.message_chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.previous_words = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)

    def fit(self, turns: list[Turn]) -> "ContextFeatures":
        messages = [t.message for t in turns]
        self.message_words.fit(messages)
        self.message_chars.fit(messages)
        previous = [t.prev_message for t in turns]
        if self.use_context and any(p.strip() for p in previous):
            self.previous_words.fit(previous)
        else:
            self.use_context = False
        return self

    def transform(self, turns: list[Turn]):
        parts = [
            self.message_words.transform([t.message for t in turns]),
            self.message_chars.transform([t.message for t in turns]),
        ]
        if self.use_context:
            parts.append(self.previous_words.transform([t.prev_message for t in turns]))
        return hstack(parts).tocsr()