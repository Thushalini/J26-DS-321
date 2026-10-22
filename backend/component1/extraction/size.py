"""Find how big the dataset is: "10k rows", "about 5,000 records", "2 million samples".

    find_sizes("The dataset has about 12k rows.") -> [("12k rows", 12000)]
    standard_size("12k rows") -> "12000 rows"

Only a number followed by a row word counts (rows, records, samples, observations, instances,
data points, entries, examples), so "500 customers" or "3 people" are never read as dataset size.
The blueprint keeps rows only; the number of features is not needed by the feasibility model.
"""
from __future__ import annotations

import re
from typing import Optional

from ..timeline.dates import NUMBER_WORDS

ROW_WORDS = r"rows|records|samples|observations|instances|data\s+points|datapoints|entries|examples|data\s+rows"
_MULTIPLIER = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "mn": 1_000_000, "million": 1_000_000,
               "mil": 1_000_000, "lakh": 100_000, "lakhs": 100_000, "b": 1_000_000_000, "billion": 1_000_000_000}
NUMBERS = {**NUMBER_WORDS, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
           "ninety": 90, "hundred": 100, "a hundred": 100}
_WORDS = "|".join(sorted((w for w in NUMBERS if w not in ("a", "an")), key=len, reverse=True))
SIZE = re.compile(
    rf"(?<![\w.,])(?P<n>\d{{1,3}}(?:,\d{{3}})+|\d+(?:\.\d+)?|{_WORDS}|a\s+few)\s*"
    rf"(?P<mult>k|thousand|million|mil|mn|m|lakhs?|billion|b)?\b\s*(?:of\s+)?(?:labelled\s+|labeled\s+|training\s+)?"
    rf"(?P<unit>{ROW_WORDS})\b",
    re.IGNORECASE,
)
_STANDARD = re.compile(r"^\s*(\d+)\s+rows\s*$")


def _rows(n: str, mult: Optional[str]) -> Optional[int]:
    n = n.lower().replace(",", "")
    if n == "a few":
        return None
    value = NUMBERS.get(n)
    if value is None:
        try:
            value = float(n)
        except ValueError:
            return None
    value *= _MULTIPLIER.get((mult or "").lower(), 1)
    rows = int(round(value))
    return rows if rows > 0 else None


def find_sizes(text: str) -> list[tuple[str, int]]:
    """(exact words, number of rows) for each size stated in the text."""
    out = []
    for m in SIZE.finditer(text or ""):
        rows = _rows(m.group("n"), m.group("mult"))
        if rows is not None:
            out.append((m.group(0), rows))
    return out


def rows_of(value: Optional[str]) -> Optional[int]:
    """The number of rows in a stored or free-text size value, or None."""
    if not value:
        return None
    if m := _STANDARD.match(value):
        return int(m[1])
    found = find_sizes(value)
    if found:
        return found[0][1]
    bare = re.fullmatch(r"\s*~?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|thousand|million|m|lakhs?)?\s*", value, re.I)
    return _rows(bare[1], bare[2]) if bare else None


def standard_size(value: str) -> str:
    """"12k rows" -> "12000 rows". A value without a recognisable number is kept as it is."""
    rows = rows_of(value)
    return f"{rows} rows" if rows is not None else " ".join((value or "").split())
