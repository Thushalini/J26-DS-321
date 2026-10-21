"""Put an accepted document suggestion into the blueprint, with its evidence.

    label, probability = accept_candidate(store, bundle, candidate, accepted_by="M-2")

Two mentions are recorded, both by the extractor so the state classifier counts them:
  1. the document itself (speaker "doc:<file>", the page and the exact words), and
  2. the team member who accepted it (speaker = their member code).
That is two different sources agreeing on one value, which is what the codebook calls a
value "confirmed ... by a document the team accepts". Then the field is judged again, exactly
as after a chat message, so a document value that disagrees with the chat shows as a conflict.
"""
from __future__ import annotations

from typing import Optional

from ..checks.plan_checks import record_checks
from ..common.events import EventStore
from ..common.schema import FieldName, FieldStatus, Provenance, SourceType
from ..pipeline import judge_field
from ..state.features import EXTRACTOR_ACTOR
from .candidates import Candidate

DOCUMENT_SPEAKER_PREFIX = "doc:"


def document_speaker(candidate: Candidate) -> str:
    return DOCUMENT_SPEAKER_PREFIX + candidate.filename


def accept_candidate(
    store: EventStore,
    bundle: dict,
    candidate: Candidate,
    accepted_by: str,
    via: Optional[str] = None,
) -> tuple[str, float]:
    """Record the suggestion as accepted. Returns the field's new (status, model confidence).

    via: where the acceptance happened, e.g. a chat message id; defaults to the Blueprint tab.
    """
    name = FieldName(candidate.field)
    document = Provenance(
        source_type=SourceType.DOCUMENT,
        source_id=candidate.doc_id,
        speaker=document_speaker(candidate),
        page=candidate.page,
        text_span=candidate.span,
    )
    store.update_field(name, candidate.value, FieldStatus.KNOWN, 0.5, document, EXTRACTOR_ACTOR)
    acceptance = Provenance(
        source_type=SourceType.DOCUMENT,
        source_id=via or f"accept:{candidate.fact_ids[0]}",
        speaker=accepted_by,
        page=candidate.page,
        text_span=f"Accepted from {candidate.source_label}",
    )
    store.update_field(name, candidate.value, FieldStatus.KNOWN, 0.5, acceptance, EXTRACTOR_ACTOR)
    result = judge_field(store, bundle, candidate.field)
    record_checks(store)              # an accepted value can make the plan fit, or stop fitting
    return result
