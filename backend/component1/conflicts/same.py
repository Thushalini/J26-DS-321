"""Are two values for one field the same choice?

    same_value("algorithm", "RF", "random forest")                -> True   (known names)
    same_value("target_variable", "tourist arrivals", "hotel occupancy") -> False  (NLI: contradiction)
    same_value("target_variable", "monthly tourist arrivals", "tourist arrivals") -> True  (NLI: implies)

Order: the same spelling; then known names for algorithm, metric, task and resources (if both
values are known names, they decide alone); then, for free-text fields, the NLI model: same only
when neither statement contradicts the other AND at least one implies the other.
"""
from __future__ import annotations

from typing import Optional

from ..state.normalize import normalize_value
from .nli import relation

KNOWN_NAME_KIND = {"algorithm": "algorithm", "evaluation_metric": "metric", "task_type": "task", "resources": "resource"}
FREE_TEXT = {"target_variable", "dataset", "problem", "objective", "domain"}
MAX_CONTRADICTION = 0.5
MIN_ENTAILMENT = 0.5


def statement(field: str, value: str) -> str:
    return f"The {field.replace('_', ' ')} of the project is {value}."


def known_names(field: str, value: str) -> list[str]:
    kind = KNOWN_NAME_KIND.get(field)
    if kind is None:
        return []
    from ..extraction.rules import canonical_names
    return canonical_names(kind, value)


def same_value(field: str, a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    if normalize_value(a) == normalize_value(b):
        return True
    names_a, names_b = known_names(field, a), known_names(field, b)
    if names_a and names_b:
        return names_a[0] == names_b[0]
    if field not in FREE_TEXT:
        return False
    one, two = statement(field, a), statement(field, b)
    forward, backward = relation(one, two), relation(two, one)
    if forward is None or backward is None:
        return False                                   # NLI off or unavailable: spelling only
    if max(forward["contradiction"], backward["contradiction"]) >= MAX_CONTRADICTION:
        return False
    return max(forward["entailment"], backward["entailment"]) >= MIN_ENTAILMENT


def group_key(field: str, value: str, seen: list[str]) -> str:
    """The key of the first earlier value that means the same, else this value's own key.
    seen: earlier values (original spellings) in order; this value is NOT added."""
    for earlier in seen:
        if same_value(field, earlier, value):
            return normalize_value(earlier)
    return normalize_value(value)
