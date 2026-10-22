"""Read one chat message and fill in the blueprint (Day 2).

The language model is passed in as a plain function (prompt in, text out).
Tests use a fake one, and the real provider is plugged in later.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Callable, Optional

from pydantic import BaseModel, Field, ValidationError

from ..common.events import EventStore
from ..common.schema import Event, FieldName, FieldStatus, Provenance, SourceType
from ..timeline.dates import MILESTONE_PREFIX
from ..timeline.find import standard_timeline_value
from ..timeline.milestones import record_milestone
from .size import standard_size

LLMFn = Callable[[str], str]  # takes a prompt, returns the model's reply text

# Placeholder until real confidence scoring is built (feature 7).
PLACEHOLDER_CONFIDENCE = 0.5

PROMPT_TEMPLATE = """You read ONE message from a team planning a data science project.
Pull out only planning facts that the message actually states.

Fields you may use:
- problem: what the project tries to solve
- domain: the area, for example tourism or health
- objective: what the team wants to achieve
- dataset: the data they will use
- dataset_size: how many rows the dataset has, for example "12000 rows"
- target_variable: what they want to predict
- task_type: for example classification, regression or forecasting
- algorithm: the model or method
- evaluation_metric: how they measure results
- resources: compute or budget
- timeline: the project's overall deadline or length, for example "8 weeks" or "deadline 15 December".
  For a NAMED milestone (demo, presentation, mid evaluation, ...) use the value
  "milestone: <name>, <date>", for example "milestone: Demo, 20 November".

Rules:
- Do not guess. If the message says nothing about a field, leave it out.
- "span" must be copied exactly from the message.
- Set "hedged" to true if the speaker sounds unsure (maybe, I think, not sure).
- Reply with JSON only, in this shape:
{"items": [{"field": "algorithm", "value": "XGBoost", "span": "use XGBoost", "hedged": false}]}
- If there is nothing to extract, reply with {"items": []}

Message:
<<<
{message}
>>>
"""


class ExtractedItem(BaseModel):
    field: FieldName
    value: str = Field(min_length=1)
    span: str = Field(min_length=1)
    hedged: bool = False


def build_prompt(message: str) -> str:
    return PROMPT_TEMPLATE.replace("{message}", message)


_MESSAGE_START = "Message:\n<<<\n"
_MESSAGE_END = "\n>>>\n"


def message_from_prompt(prompt: str) -> str:
    """The message that build_prompt() wrapped. Lets a non-LLM extractor stand in for the LLM."""
    start = prompt.find(_MESSAGE_START)
    end = prompt.rfind(_MESSAGE_END)
    if start == -1 or end == -1 or end < start + len(_MESSAGE_START):
        return prompt
    return prompt[start + len(_MESSAGE_START) : end]


def _parse_json(raw: str) -> Optional[dict]:
    """Find the JSON object in the model's reply, even if it is wrapped in a code fence."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def extract_items(message: str, llm: LLMFn) -> list[ExtractedItem]:
    """Ask the model what the message states. Bad or invented answers are dropped."""
    data = _parse_json(llm(build_prompt(message)))
    if not data or not isinstance(data.get("items"), list):
        return []
    kept: list[ExtractedItem] = []
    for raw in data["items"]:
        if not isinstance(raw, dict):
            continue
        try:
            item = ExtractedItem.model_validate(raw)  # unknown field names fail here
        except ValidationError:
            continue
        if not item.value.strip():
            continue
        if item.span not in message:  # the quoted words must really be in the message
            continue
        if item.field == FieldName.TIMELINE:       # "eight weeks" and "8 weeks" must be one value
            item = item.model_copy(update={"value": standard_timeline_value(item.value)})
        elif item.field == FieldName.DATASET_SIZE:  # "12k rows" and "12,000 records" must be one value
            item = item.model_copy(update={"value": standard_size(item.value)})
        kept.append(item)
    return kept


def apply_message(
    store: EventStore,
    message_id: str,
    speaker: str,
    message: str,
    llm: LLMFn,
    timestamp: Optional[datetime] = None,
) -> list[Event]:
    """Extract facts from one message and write them into the blueprint.

    Returns the field updates only. Named milestones are recorded as timeline decisions instead
    (timeline/milestones.py), so they never compete with the overall deadline.
    """
    events: list[Event] = []
    for item in extract_items(message, llm):
        if item.field == FieldName.TIMELINE and item.value.lower().startswith(MILESTONE_PREFIX):
            record_milestone(store, item.value, item.span, speaker, message_id)
            continue
        provenance = Provenance(
            source_type=SourceType.CHAT,
            source_id=message_id,
            speaker=speaker,
            timestamp=timestamp,
            text_span=item.span,
        )
        status = FieldStatus.UNCERTAIN if item.hedged else FieldStatus.KNOWN
        events.append(
            store.update_field(
                item.field, item.value.strip(), status, PLACEHOLDER_CONFIDENCE, provenance, "extractor"
            )
        )
    return events