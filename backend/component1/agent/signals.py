"""What the pipeline tells Buddy. One Signal is one thing the pipeline noticed.

The pipeline builds Signals and Buddy's speaking rules (agent/policy.py) decide
which of them, if any, become a chat message. Keeping them in their own file means
the pipeline and Buddy only share this small, simple shape.
"""
from __future__ import annotations

from dataclasses import dataclass

SAVED = "saved"          # a field was just saved or changed
CONFLICT = "conflict"    # two sources disagree about a field
CHECK = "check"          # a check found a possible problem (compatibility, dataset fit, feasibility)
FOUND = "found"          # a candidate value was found in the uploaded documents
GAP = "gap"              # a field is still missing and ready to be asked about
FINALIZED = "finalized"  # a decision was just finalized (a calm moment)
CONFIRM = "confirm"      # a field likely to be disputed or changed soon: ask the team to confirm it
CLARIFY = "clarify"      # a field whose value is too vague to build on: ask for the detail


@dataclass(frozen=True)
class Signal:
    """Something the pipeline noticed. For gaps, pass them most important first.

    values: the value(s) involved. One value for SAVED and FOUND, two or more for CONFLICT.
    detail: SAVED -> a short note such as "marked uncertain". CHECK -> what the problem is.
            CONFIRM -> why the field is at risk. CLARIFY -> the question to ask. GAP -> why it matters.
    source: where it came from, e.g. "Member_A", "Proposal.pdf p.4" or "rule table".
    priority: for GAP, CONFIRM and CLARIFY, how worth asking it is (active/priority.py); 0 = not scored.
    """
    kind: str
    field: str
    values: tuple[str, ...] = ()
    detail: str = ""
    source: str = ""
    priority: float = 0.0

    @property
    def key(self) -> tuple:
        detail = self.detail if self.kind in (SAVED, CHECK) else ""
        return (self.kind, self.field, "|".join(sorted(self.values)), detail)