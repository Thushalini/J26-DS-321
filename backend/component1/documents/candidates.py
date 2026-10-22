"""Turn document facts into suggestions: per field, the values the documents give, best first.

A suggestion groups every OPEN fact with the same field and the same value (after the same
spelling clean-up the state classifier uses, so "XGBoost" and "xgboost" are one suggestion).
The best one per field is the value mentioned most often; ties go to the earliest mention.
A value the team dismissed is never suggested again, even if another document repeats it.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Iterable, Optional

from ..common.schema import FieldName
from ..state.normalize import normalize_value


@dataclass
class Candidate:
    field: str
    value: str                 # the spelling of the first mention
    doc_id: str                # where the first mention is
    filename: str
    page: Optional[int]
    span: str                  # the exact words of the first mention
    mentions: int = 1
    fact_ids: list[str] = dc_field(default_factory=list)
    documents: set[str] = dc_field(default_factory=set)

    @property
    def source_label(self) -> str:
        """e.g. "Proposal.pdf p.4" or "Notes.docx"."""
        return f"{self.filename} p.{self.page}" if self.page else self.filename

    @property
    def key(self) -> str:
        return normalize_value(self.value)

    def as_dict(self, include_evidence: bool = True) -> dict:
        out = {
            "fact_id": self.fact_ids[0],
            "field": self.field,
            "value": self.value,
            "mentions": self.mentions,
            "documents": len(self.documents),
            "source": self.source_label,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page": self.page,
        }
        if include_evidence:
            out["text"] = self.span
        return out


def group_candidates(facts: Iterable[dict]) -> dict[str, list[Candidate]]:
    """field -> its open suggestions, best first. facts are rows from storage.list_doc_facts."""
    facts = list(facts)
    dismissed = {(f["field"], normalize_value(f["value"])) for f in facts if f["status"] == "dismissed"}
    decided = {(f["field"], normalize_value(f["value"])) for f in facts if f["status"] == "accepted"}

    groups: dict[tuple[str, str], Candidate] = {}
    order: list[tuple[str, str]] = []
    for f in facts:                    # rows arrive oldest document first, then page and chunk
        if f["status"] != "open":
            continue
        key = (f["field"], normalize_value(f["value"]))
        if not key[1] or key in dismissed or key in decided:
            continue
        if key not in groups:
            groups[key] = Candidate(
                field=f["field"], value=f["value"], doc_id=f["doc_id"], filename=f.get("filename") or "document",
                page=f.get("page"), span=f["span"], mentions=0,
            )
            order.append(key)
        c = groups[key]
        c.mentions += 1
        c.fact_ids.append(f["fact_id"])
        c.documents.add(f["doc_id"])

    by_field: dict[str, list[Candidate]] = {}
    for rank, key in enumerate(order):
        by_field.setdefault(key[0], []).append(groups[key])
    first_seen = {key: i for i, key in enumerate(order)}
    for name, items in by_field.items():
        items.sort(key=lambda c: (-c.mentions, first_seen[(c.field, c.key)]))
    # Keep the fields in the blueprint's own order, so screens and Buddy are predictable.
    return {n.value: by_field[n.value] for n in FieldName if n.value in by_field}


def best_candidates(facts: Iterable[dict]) -> dict[str, Candidate]:
    """field -> its single best open suggestion."""
    return {name: items[0] for name, items in group_candidates(facts).items()}


def find_group(facts: Iterable[dict], fact_id: str) -> Optional[Candidate]:
    """The open suggestion that contains this fact, or None if it is not open any more."""
    for items in group_candidates(facts).values():
        for c in items:
            if fact_id in c.fact_ids:
                return c
    return None
