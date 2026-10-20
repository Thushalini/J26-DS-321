"""Make two spellings of the same value compare as equal (feature 3 support).

"XGBoost", "xgboost " and "The XGBoost." must count as ONE value, or the system
would invent conflicts. Keep the original text for display and use this only to compare.

Known limits (say so in the report): it does not know synonyms or abbreviations
("RF" vs "random forest") or hyphenation ("random-forest" vs "random forest").
Those need the conflict model (feature 10).
"""
from __future__ import annotations

import re
import unicodedata

_ARTICLES = ("the ", "an ", "a ")
_EDGE_PUNCTUATION = " \t.,;:!?\"'()[]{}"


def normalize_value(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"\s+", " ", t).strip(_EDGE_PUNCTUATION)
    for article in _ARTICLES:
        if t.startswith(article):
            t = t[len(article):].strip(_EDGE_PUNCTUATION)
            break
    return t