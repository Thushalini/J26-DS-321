"""Make two spellings of the same value compare as equal (feature 3 support).

"XGBoost", "xgboost " and "The XGBoost." must count as ONE value, or the system
would invent conflicts. Keep the original text for display and use this only to compare.

It also handles these everyday differences:
  - a leading label: "dataset as sri lankan tourism dataset" is "sri lankan tourism dataset"
  - underscores and hyphens: "tourist_arrival" and "random-forest" are written with spaces
  - a plural "s": "tourist arrivals" and "tourist arrival" are the same value

Known limits (say so in the report): it does not know synonyms or abbreviations
("RF" vs "random forest"). The plural rule is crude (it only drops a final "s"), and the
only word variant handled is the small list in _WORD_VARIANTS, a stop-gap.
Everything else needs the conflict model (feature 10).
"""
from __future__ import annotations

import re
import unicodedata

_ARTICLES = ("the ", "an ", "a ")
_EDGE_PUNCTUATION = " \t.,;:!?\"'()[]{}"

# A leading label like "dataset as", "target variable is", "metric: ".
_LEADING_LABEL = re.compile(
    r"^(?:data ?set|target variable|target|algorithm|model|evaluation metric|metric|problem|domain)"
    r"\s*(?:as|is|=|:)\s+"
)

# Words written two ways for the same thing. Kept tiny on purpose.
_WORD_VARIANTS = {"lankan": "lanka"}


def _word(w: str) -> str:
    w = _WORD_VARIANTS.get(w, w)
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def normalize_value(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[_\-]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip(_EDGE_PUNCTUATION)
    t = _LEADING_LABEL.sub("", t).strip(_EDGE_PUNCTUATION)
    for article in _ARTICLES:
        if t.startswith(article):
            t = t[len(article):].strip(_EDGE_PUNCTUATION)
            break
    return " ".join(_word(w) for w in t.split(" "))