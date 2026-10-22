"""Domino validation: when an earlier choice changes, flag the later decisions it may break.

    changes = run_domino(store, changed={"task_type": ("classification", "regression")})
    restored = restore_passing(store)

The plan's order comes from DEPENDS_ON (problem -> dataset -> target -> task type -> algorithm and
metric -> resources). When a field's value changes, every settled (confirmed or finalized) decision
further down is looked at:
  * if a plan check covers it (metric, algorithm, task type, resources) and now fails or warns, the
    decision is set to pending_fix, with a reason that names the change and the check's advice;
  * if no check covers it (domain, objective, dataset, target, ...) and it was FINALIZED on a field
    directly below the change, it is set to pending_fix too: "re-check it", because nothing can
    tell automatically whether it still holds;
  * if the check still passes, nothing happens: the change did not break it.
A pending_fix decision goes back to its old status by itself once its check passes again, or when
someone agrees with it again, or when the team settles a new value for the field.
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..checks.plan_checks import run_checks
from ..common.events import EventStore
from ..common.schema import DEPENDS_ON, DecisionStatus, FieldName
from .tracker import ACTOR, SETTLED, Change, active_for, effective_state

CHECKED_FIELDS = {"evaluation_metric", "algorithm", "task_type", "resources"}


def label(field: str) -> str:
    return field.replace("_", " ")


def descendants(field: str) -> list[str]:
    """Every field that depends on this one, directly or through others, nearest first."""
    children: dict[str, list[str]] = {}
    for child, parents in DEPENDS_ON.items():
        for parent in parents:
            children.setdefault(parent.value, []).append(child.value)
    out, queue = [], list(children.get(field, []))
    while queue:
        f = queue.pop(0)
        if f not in out:
            out.append(f)
            queue.extend(children.get(f, []))
    return out


def direct_children(field: str) -> set[str]:
    return {child.value for child, parents in DEPENDS_ON.items() if FieldName(field) in parents}


def _failing(results, field: str):
    return [r for r in results if r.field == field and r.status != "pass"]


def run_domino(store: EventStore, changed: dict[str, tuple[Optional[str], Optional[str]]],
               results=None) -> list[Change]:
    """changed: field -> (old value, new value), for fields whose value just changed.
    results: plan check results for the new state (computed if not given)."""
    if not changed:
        return []
    results = results if results is not None else run_checks(effective_state(store.state), store.events)
    out: list[Change] = []
    for upstream, (old, new) in changed.items():
        if not old or not new:
            continue                                     # a first value breaks nothing
        for field in descendants(upstream):
            for d in active_for(store.state, field):
                if d.status not in SETTLED:
                    continue
                failing = _failing(results, field) if field in CHECKED_FIELDS else []
                cause = f"the {label(upstream)} changed from {old} to {new}"
                if failing:
                    reason = f"{cause}, so the {label(field)} '{d.value}' may no longer fit: {failing[0].detail}"
                elif (field not in CHECKED_FIELDS and field in direct_children(upstream)
                      and d.status == DecisionStatus.FINALIZED):
                    reason = (f"{cause}, and the {label(field)} '{d.value}' was finalized before that. "
                              f"It may no longer be right; please re-check it.")
                else:
                    continue
                store.change_decision_status(d.decision_id, DecisionStatus.PENDING_FIX, ACTOR,
                                             pending_reason=reason, held_status=d.status)
                out.append(Change(d.decision_id, field, d.value, "pending_fix", f"change of {label(upstream)}"))
    return out


def restore_passing(store: EventStore, results=None) -> list[Change]:
    """pending_fix decisions whose plan check passes again go back to their old status."""
    results = results if results is not None else run_checks(effective_state(store.state), store.events)
    out = []
    for d in [d for d in store.state.decisions.values() if d.status == DecisionStatus.PENDING_FIX]:
        field = d.field.value
        if field not in CHECKED_FIELDS or _failing(results, field):
            continue                                     # unchecked fields need a person to re-confirm
        back = d.held_status or DecisionStatus.CONFIRMED
        store.change_decision_status(d.decision_id, back, ACTOR, pending_reason=None, held_status=None)
        out.append(Change(d.decision_id, field, d.value, "restored", "plan check"))
    return out


def values_of(state, fields: Iterable[FieldName] = FieldName) -> dict[str, Optional[str]]:
    return {f.value: state.fields[f].value for f in fields}


def changed_values(before: dict[str, Optional[str]], after: dict[str, Optional[str]]) -> dict:
    """Fields whose value changed in meaning (conflicts/same.py: spelling, known names, NLI)."""
    from ..conflicts.same import same_value
    out = {}
    for f, old in before.items():
        new = after.get(f)
        if old and new and not same_value(f, old, new):
            out[f] = (old, new)
    return out


def settle(store: EventStore, changes: Iterable[Change]) -> list[Change]:
    """After decision changes: run Domino validation for every field whose settled value was just
    replaced, and give back their old status to re-checks whose plan check passes again."""
    from .tracker import current_decision

    changed: dict[str, tuple[str, str]] = {}
    for c in changes:
        if c.kind != "superseded":
            continue
        new = current_decision(store.state, c.field)
        if new is not None and new.status in SETTLED and new.value != c.value:
            changed[c.field] = (c.value, new.value)
    results = run_checks(effective_state(store.state), store.events)
    out = run_domino(store, changed, results)
    out += restore_passing(store, results)
    return out
