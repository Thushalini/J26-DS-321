"""Turn timeline values into dates.

Values are stored as short standard text, so the same plan written two ways is one value:
    "8 weeks"                     a project length ("eight weeks", "an 8-week project")
    "deadline 2026-12-15"         a full date
    "deadline 15 December"        a date without a year (the next 15 December after it was said)
    "deadline next friday"        a relative date (worked out from when it was said)
    "deadline week 10"            a teaching week; no calendar date can be worked out
    "milestone: Demo, 2026-11-20" a named milestone (see find.py)

resolve(value, said_on) gives the dates. Dates written with slashes are read day first
(15/12/2026 = 15 December), as in Sri Lanka and the UK. Set C1_DATES_MONTH_FIRST=1 for 12/15/2026.
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from typing import Optional

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "twenty": 20,
}
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

MILESTONE_PREFIX = "milestone: "
DEADLINE_PREFIX = "deadline "

_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_NUMERIC = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2}|\d{4})$")
_DAY_MONTH = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)\.?(?:,?\s+(\d{4}))?$")
_MONTH_DAY = re.compile(r"^([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?$")
_DURATION = re.compile(r"^(\d{1,3}|[a-z]+)[\s-]+(day|week|month|year)s?$")
_WEEK_NO = re.compile(r"^week\s+(\d{1,2})$")


def month_first() -> bool:
    return (os.environ.get("C1_DATES_MONTH_FIRST") or "").strip().lower() in ("1", "true", "yes")


def number(word: str) -> Optional[int]:
    word = word.lower()
    if word.isdigit():
        return int(word)
    return NUMBER_WORDS.get(word)


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _next_on_or_after(month: int, day: int, ref: date) -> Optional[date]:
    """The first month/day on or after ref (a date said without a year)."""
    for year in (ref.year, ref.year + 1):
        d = _safe_date(year, month, day)
        if d is not None and d >= ref:
            return d
    return None


def parse_date(text: str, ref: Optional[date] = None) -> Optional[date]:
    """A calendar date from a short phrase, or None. ref is needed for dates without a year and for
    relative dates ("next friday", "tomorrow")."""
    t = " ".join(text.lower().replace(",", " ").split())
    if m := _ISO.match(t):
        return _safe_date(int(m[1]), int(m[2]), int(m[3]))
    if m := _NUMERIC.match(t):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        y = y + 2000 if y < 100 else y
        day, month = (b, a) if month_first() else (a, b)
        return _safe_date(y, month, day)
    for pattern, day_group, month_group in ((_DAY_MONTH, 1, 2), (_MONTH_DAY, 2, 1)):
        if (m := pattern.match(t)) and m[month_group] in MONTHS:
            month, day = MONTHS[m[month_group]], int(m[day_group])
            if m[3]:
                return _safe_date(int(m[3]), month, day)
            return _next_on_or_after(month, day, ref) if ref else None
    if ref is None:
        return None
    if t == "today":
        return ref
    if t == "tomorrow":
        return ref + timedelta(days=1)
    if m := re.fullmatch(r"in\s+(\d{1,3}|[a-z]+)\s+(day|week|month|year)s?", t):
        n = number(m[1])
        return ref + timedelta(days=n * UNIT_DAYS[m[2]]) if n else None
    words = t.split()
    if words and words[-1] in WEEKDAYS:
        ahead = (WEEKDAYS[words[-1]] - ref.weekday()) % 7
        if len(words) == 2 and words[0] == "next":
            ahead = ahead or 7
        elif len(words) == 1 or words[0] in ("this", "on", "by", "coming"):
            ahead = ahead or 7
        else:
            return None
        return ref + timedelta(days=ahead)
    return None


def duration_days(text: str) -> Optional[int]:
    """'8 weeks' -> 56. None if the text is not a length of time."""
    m = _DURATION.match(" ".join(text.lower().split()))
    if not m:
        return None
    n = number(m[1])
    return None if n is None or n <= 0 else n * UNIT_DAYS[m[2]]


def standard_duration(n: int, unit: str) -> str:
    unit = unit.lower().rstrip("s")
    return f"{n} {unit}" + ("" if n == 1 else "s")


def standard_date_text(text: str) -> str:
    """A date phrase in standard form: full dates become ISO, everything else is tidied lower case."""
    text = text.strip().rstrip(".!?;:")
    d = parse_date(text)                       # without ref: only dates that carry their own year
    if d is not None:
        return d.isoformat()
    return " ".join(text.lower().replace(",", " ").split())


def _as_date(said_on) -> Optional[date]:
    if said_on is None:
        return None
    if isinstance(said_on, datetime):
        return said_on.date()
    if isinstance(said_on, date):
        return said_on
    try:
        return datetime.fromisoformat(str(said_on).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def resolve(value: Optional[str], said_on=None, today: Optional[date] = None) -> Optional[dict]:
    """What a stored timeline value means in dates. None if there is no value.

    said_on: when the value was first said (a date, datetime or ISO text); needed for lengths
    ("8 weeks" ends 8 weeks after it was said) and for relative or year-less dates.
    """
    if not value:
        return None
    ref, today = _as_date(said_on), today or date.today()
    text = value.strip()
    out = {"kind": "text", "text": text, "deadline": None, "duration_days": None, "days_left": None, "week": None}
    if text.lower().startswith(MILESTONE_PREFIX):
        return None                                   # milestones are listed separately
    if (days := duration_days(text)) is not None:
        out.update(kind="duration", duration_days=days)
        if ref is not None:
            out["deadline"] = (ref + timedelta(days=days)).isoformat()
    elif text.lower().startswith(DEADLINE_PREFIX):
        rest = text[len(DEADLINE_PREFIX):]
        if m := _WEEK_NO.match(rest.lower().strip()):
            out.update(kind="week", week=int(m[1]))
        else:
            d = parse_date(rest, ref)
            out["kind"] = "deadline"
            if d is not None:
                out["deadline"] = d.isoformat()
    if out["deadline"]:
        out["days_left"] = (date.fromisoformat(out["deadline"]) - today).days
    return out


def resolve_milestone(value: str, said_on=None) -> Optional[dict]:
    """'milestone: Demo, 2026-11-20' -> {'name': 'Demo', 'date': '2026-11-20', 'when': '2026-11-20'}."""
    if not value or not value.lower().startswith(MILESTONE_PREFIX):
        return None
    body = value[len(MILESTONE_PREFIX):]
    name, _, when = body.partition(",")
    when = when.strip()
    m = _WEEK_NO.match(when.lower())
    d = None if m else parse_date(when, _as_date(said_on))
    return {"name": name.strip(), "when": when, "date": d.isoformat() if d else None,
            "week": int(m[1]) if m else None}
