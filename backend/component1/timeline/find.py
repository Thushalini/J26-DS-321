"""Find timeline phrases in a sentence. Only phrases with a planning cue count, so "see you in 2
weeks" or "meeting tomorrow" are never read as the project's timeline.

    find_timeline("Submission is on 15 December and the demo is on 20/11/2026.")
    -> overall:    [Found("Submission is on 15 December", "deadline 15 december")]
       milestones: [Found("demo is on 20/11/2026", "milestone: Demo, 2026-11-20")]

overall    the project's end: a deadline ("due on", "submission", "final evaluation") or a length
           ("we have 8 weeks", "an 8-week project", "8 weeks left")
milestones named steps before the end: demo, presentation, mid evaluation, progress review, ...
Every span is copied exactly from the sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .dates import DEADLINE_PREFIX, MILESTONE_PREFIX, MONTHS, NUMBER_WORDS, WEEKDAYS, number, standard_date_text, \
    standard_duration

_MONTH_NAMES = "|".join(sorted(MONTHS, key=len, reverse=True))
_WEEKDAY_NAMES = "|".join(WEEKDAYS)
_NUMBER_NAMES = "|".join(sorted((w for w in NUMBER_WORDS if w not in ("a", "an")), key=len, reverse=True))

DATE = re.compile(
    rf"(?<![\w/.-])(?:"
    rf"\d{{4}}-\d{{1,2}}-\d{{1,2}}"
    rf"|\d{{1,2}}[/.-]\d{{1,2}}[/.-](?:\d{{4}}|\d{{2}})"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTH_NAMES})(?:,?\s+\d{{4}})?"
    rf"|(?:{_MONTH_NAMES})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|(?:next|this|coming)\s+(?:{_WEEKDAY_NAMES})"
    rf"|(?<=on\s)(?:{_WEEKDAY_NAMES})|(?<=by\s)(?:{_WEEKDAY_NAMES})"
    rf"|tomorrow"
    rf"|week\s+\d{{1,2}}"
    rf"|in\s+(?:\d{{1,3}}|{_NUMBER_NAMES}|a|an)\s+(?:day|week|month)s?"
    rf")(?![\w/])",
    re.IGNORECASE,
)

# cue -> standard milestone name
MILESTONES = {
    r"mid[- ]?(?:term\s+)?evaluation|mid[- ]?eval": "Mid evaluation",
    r"progress\s+(?:review|presentation)": "Progress review",
    r"proposal\s+(?:presentation|defen[cs]e)": "Proposal presentation",
    r"final\s+presentation": "Final presentation",
    r"interim\s+(?:report|review)": "Interim review",
    r"sprint\s+review": "Sprint review",
    r"demo(?:nstration)?": "Demo",
    r"presentation": "Presentation",
    r"viva": "Viva",
    r"prototype": "Prototype",
    r"milestone": "Milestone",
}
DEADLINE_CUES = (r"deadline|due|submission|submit(?:ted)?|hand(?:ed)?\s+in|final\s+(?:evaluation|report)|"
                 r"project\s+ends?|finish(?:ed)?|complete(?:d)?|wrap\s+up")
_MILESTONE_CUE = re.compile(r"\b(?:" + "|".join(MILESTONES) + r")\b", re.IGNORECASE)
_DEADLINE_CUE = re.compile(rf"\b(?:{DEADLINE_CUES})\b", re.IGNORECASE)

_N = rf"(?:\d{{1,3}}|{_NUMBER_NAMES}|a|an)"
DURATION = [
    # "we have 8 weeks", "we've got two months", "the project lasts 14 weeks", "timeline is 8 weeks"
    re.compile(rf"\b(?:we\s+have|we've\s+got|we\s+got|we\s+only\s+have|project\s+(?:is|lasts|takes|runs(?:\s+for)?)|"
               rf"timeline\s+is|duration\s+is|time\s?frame\s+is|length\s+is)\s+(?:about\s+|around\s+|only\s+)?"
               rf"(?P<n>{_N})[\s-]+(?P<u>day|week|month|year)s?\b", re.IGNORECASE),
    # "an 8-week project", "a 3 month timeline"
    re.compile(rf"\b(?P<n>{_N})[\s-]+(?P<u>day|week|month|year)s?[\s-]+(?:long\s+)?(?:project|timeline|plan|"
               rf"programme|program|sprint\s+plan)\b", re.IGNORECASE),
    # "8 weeks left", "two months to go", "6 weeks remaining"
    re.compile(rf"\b(?P<n>{_N})[\s-]+(?P<u>day|week|month|year)s?\s+(?:left|to\s+go|remaining)\b", re.IGNORECASE),
]
_MAX_GAP = 60      # characters between a cue and its date
_NOT_BEFORE = re.compile(r"\b(?:not|isn't|isnt|no\s+longer)\s+(?:on\s+|by\s+|the\s+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class Found:
    span: str
    value: str


def _nearest_cue(sentence: str, date_start: int, date_end: int):
    """The closest milestone or deadline cue to a date: ("milestone", name, start) or ("deadline", None, start)."""
    best = None
    for kind, pattern in (("milestone", _MILESTONE_CUE), ("deadline", _DEADLINE_CUE)):
        for m in pattern.finditer(sentence):
            if m.end() <= date_start:
                gap = date_start - m.end()
            elif m.start() >= date_end:
                gap = m.start() - date_end + 5          # a cue after the date is a little weaker
            else:
                continue
            if gap > _MAX_GAP:
                continue
            name = None
            if kind == "milestone":
                name = next(n for p, n in MILESTONES.items() if re.fullmatch(p, m.group(0), re.IGNORECASE))
            if best is None or gap < best[0]:
                best = (gap, kind, name, m.start(), m.end())
    return best


def find_timeline(sentence: str) -> tuple[list[Found], list[Found]]:
    """(overall, milestones) found in one sentence."""
    overall: list[Found] = []
    milestones: list[Found] = []
    taken: list[tuple[int, int]] = []

    for m in DATE.finditer(sentence):
        if _NOT_BEFORE.search(sentence[max(0, m.start() - 15):m.start()]):     # "not 15 Dec, it's 20 Dec"
            continue
        cue = _nearest_cue(sentence, m.start(), m.end())
        if cue is None:
            continue
        _, kind, name, cue_start, cue_end = cue
        start, end = min(cue_start, m.start()), max(cue_end, m.end())
        when = standard_date_text(m.group(0))
        if when.startswith("in "):                    # "in 3 weeks" is only a date for a deadline
            if kind != "deadline":
                continue
        span = sentence[start:end]
        if kind == "milestone":
            milestones.append(Found(span, f"{MILESTONE_PREFIX}{name}, {when}"))
        else:
            overall.append(Found(span, f"{DEADLINE_PREFIX}{when}"))
        taken.append((m.start(), m.end()))

    for pattern in DURATION:
        for m in pattern.finditer(sentence):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            n = number(m.group("n"))
            if not n:
                continue
            overall.append(Found(m.group(0), standard_duration(n, m.group("u"))))
            taken.append((m.start(), m.end()))

    overall.sort(key=lambda f: sentence.find(f.span))
    return overall, milestones


def standard_timeline_value(value: str) -> str:
    """Bring a timeline value (for example from the language model) into the standard form."""
    from .dates import duration_days, parse_date

    v = " ".join((value or "").split())
    low = v.lower()
    if low.startswith(MILESTONE_PREFIX):
        return v
    m = re.fullmatch(r"(\d{1,3}|[a-z]+)[\s-]+(day|week|month|year)s?", low)
    if m and duration_days(low) is not None:
        return standard_duration(number(m[1]), m[2])
    if low.startswith(DEADLINE_PREFIX):
        return DEADLINE_PREFIX + standard_date_text(v[len(DEADLINE_PREFIX):])
    if parse_date(low) is not None or DATE.fullmatch(v):
        return DEADLINE_PREFIX + standard_date_text(v)
    return v
