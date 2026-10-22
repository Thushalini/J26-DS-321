"""Data for the decision-intent classifier (feature 8).

One row is one chat message with its label, inside a conversation:

    conv_id, turn, speaker, message, intent

The model also sees the PREVIOUS message of the same conversation, because "I agree" only
means something next to what it answers. Train and test sets are always split by whole
conversation, never by single message, so no conversation leaks from one side to the other.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

LABELS = ["proposal", "agreement", "finalization", "rejection", "other"]


@dataclass(frozen=True)
class Turn:
    conv_id: str
    turn: int
    speaker: str
    message: str
    prev_message: str
    intent: str


def load_turns(path, label_col: str = "intent") -> list[Turn]:
    """Read a labeled conversations file. Raises a clear error for a missing column or label."""
    with open(Path(path), encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = {"conv_id", "turn", "speaker", "message", label_col} - columns
        if missing:
            raise ValueError(f"{path} is missing the column(s): {', '.join(sorted(missing))}")
        rows = list(reader)

    by_conv: dict[str, list[dict]] = {}
    for r in rows:
        by_conv.setdefault(r["conv_id"], []).append(r)

    turns: list[Turn] = []
    for conv_id, items in by_conv.items():
        items.sort(key=lambda r: int(r["turn"]))
        prev = ""
        for r in items:
            label = (r[label_col] or "").strip().lower()
            if label not in LABELS:
                raise ValueError(
                    f"{path}: conversation {conv_id}, turn {r['turn']} has the label {label!r}. "
                    f"Allowed labels: {', '.join(LABELS)}"
                )
            message = (r["message"] or "").strip()
            turns.append(Turn(conv_id, int(r["turn"]), r["speaker"], message, prev, label))
            prev = message
    return turns


def group_split(turns: list[Turn], test_fraction: float = 0.2, seed: int = 42):
    """Split by whole conversation. Returns (train_turns, test_turns)."""
    conversations = sorted({t.conv_id for t in turns})
    random.Random(seed).shuffle(conversations)
    n_test = max(1, round(len(conversations) * test_fraction))
    test_ids = set(conversations[:n_test])
    return [t for t in turns if t.conv_id not in test_ids], [t for t in turns if t.conv_id in test_ids]