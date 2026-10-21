"""Blueprint schema for Component 1 (Pydantic v2).

This file defines WHAT the Living DS Project Blueprint looks like.
The event log in events.py defines HOW it changes over time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- enums
class FieldName(str, Enum):
    """The planning fields Component 1 extracts and tracks."""
    DOMAIN = "domain"
    PROBLEM = "problem"
    OBJECTIVE = "objective"
    DATASET = "dataset"
    TARGET_VARIABLE = "target_variable"
    TASK_TYPE = "task_type"
    ALGORITHM = "algorithm"
    EVALUATION_METRIC = "evaluation_metric"
    RESOURCES = "resources"
    TIMELINE = "timeline"      # the overall deadline or project length; named milestones are decisions
    DATASET_SIZE = "dataset_size"   # how many rows the dataset has, e.g. "10000 rows"


class FieldStatus(str, Enum):
    """How reliable the current value of a field is."""
    MISSING = "missing"
    KNOWN = "known"
    UNCERTAIN = "uncertain"
    CONFLICTING = "conflicting"
    CONFIRMED = "confirmed"


class DecisionStatus(str, Enum):
    """Where a decision is in its life."""
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    PENDING_FIX = "pending_fix"  # set by Domino validation when a change breaks a downstream field


class SourceType(str, Enum):
    CHAT = "chat"
    DOCUMENT = "document"
    EXTERNAL = "external"


# ------------------------------------------- gap-detection configuration
# STARTING VALUES. They follow the usual DS lifecycle order (cite CRISP-DM
# in the report). They are hand-set, so say so in the report.
DEPENDS_ON: dict[FieldName, list[FieldName]] = {
    FieldName.PROBLEM: [],
    FieldName.DOMAIN: [FieldName.PROBLEM],
    FieldName.OBJECTIVE: [FieldName.PROBLEM],
    FieldName.DATASET: [FieldName.PROBLEM],
    FieldName.TARGET_VARIABLE: [FieldName.DATASET],
    FieldName.TASK_TYPE: [FieldName.TARGET_VARIABLE],
    FieldName.ALGORITHM: [FieldName.TASK_TYPE],
    FieldName.EVALUATION_METRIC: [FieldName.TASK_TYPE],
    FieldName.RESOURCES: [FieldName.ALGORITHM],
    FieldName.TIMELINE: [FieldName.PROBLEM],   # worth asking once the team knows what it is building
    FieldName.DATASET_SIZE: [FieldName.DATASET],
}

BASE_WEIGHT: dict[FieldName, int] = {
    FieldName.PROBLEM: 10,
    FieldName.OBJECTIVE: 9,
    FieldName.DATASET: 9,
    FieldName.TARGET_VARIABLE: 9,
    FieldName.DOMAIN: 8,
    FieldName.TASK_TYPE: 8,
    FieldName.ALGORITHM: 6,
    FieldName.EVALUATION_METRIC: 6,
    FieldName.TIMELINE: 5,                     # after the core plan, before compute resources
    FieldName.DATASET_SIZE: 4,                 # needed for the feasibility check
    FieldName.RESOURCES: 3,
}


# --------------------------------------------------------------- models
class Provenance(BaseModel):
    """Where a value came from."""
    source_type: SourceType
    source_id: str                      # message id or document name
    speaker: Optional[str] = None       # chat only
    timestamp: Optional[datetime] = None
    page: Optional[int] = Field(default=None, ge=1)  # documents only
    text_span: str = Field(min_length=1)  # the exact words the value came from


class FieldState(BaseModel):
    """The current state of one planning field."""
    name: FieldName
    value: Optional[str] = None
    status: FieldStatus = FieldStatus.MISSING
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance: list[Provenance] = Field(default_factory=list)
    updated_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _value_matches_status(self) -> "FieldState":
        if self.status == FieldStatus.MISSING and self.value is not None:
            raise ValueError("a field with status 'missing' cannot have a value")
        if self.status != FieldStatus.MISSING and self.value is None:
            raise ValueError("a field with a status other than 'missing' needs a value")
        return self


class Decision(BaseModel):
    """One planning decision, with its history of who and why."""
    decision_id: str
    field: FieldName
    value: str
    status: DecisionStatus = DecisionStatus.PROPOSED
    proposed_by: str
    agreed_by: list[str] = Field(default_factory=list)
    rationale: Optional[str] = None
    supersedes: Optional[str] = None    # decision_id this one replaces
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


def _empty_fields() -> dict[FieldName, FieldState]:
    return {name: FieldState(name=name) for name in FieldName}


class Blueprint(BaseModel):
    """The Living DS Project Blueprint: the current state of a project's plan."""
    project_id: str
    version: int = 0
    fields: dict[FieldName, FieldState] = Field(default_factory=_empty_fields)
    decisions: dict[str, Decision] = Field(default_factory=dict)
    novelty_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    feasibility: Optional[dict[str, str]] = None  # constraint -> pass/warning/fail
    updated_at: Optional[datetime] = None


# --------------------------------------------------------------- events
class EventType(str, Enum):
    FIELD_UPDATED = "field_updated"
    DECISION_RECORDED = "decision_recorded"
    DECISION_STATUS_CHANGED = "decision_status_changed"
    SCORES_UPDATED = "scores_updated"


class Event(BaseModel):
    """One immutable change to a blueprint. The blueprint is rebuilt from these."""
    project_id: str
    version: int = Field(ge=1)
    event_type: EventType
    payload: dict[str, Any]
    actor: str                          # who or what made the change, e.g. "extractor"
    created_at: datetime = Field(default_factory=utcnow)