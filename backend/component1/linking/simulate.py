"""Simulated planning chats where every reply's true parent proposal is known.

Each chat has 2-5 people discussing several fields at once. Proposals name a value; replies
agree, reject or finalize. Most replies answer the newest proposal, but not all: people come back
to an older topic ("for the metric, agreed"), answer something they missed, or reply minutes
later after other messages. Replies sometimes name the value ("yes XGBoost") or the field ("the
metric is fine"), and often say nothing specific ("agreed"). Small talk is mixed in. So "the
latest proposal" is a fair but beatable baseline.

These are hand-designed patterns (the team chose simulated data over the Ubuntu IRC corpus); the
probabilities below are assumptions, stated here so the report can list them.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .features import Message

VALUES = {
    "algorithm": ["XGBoost", "Random Forest", "LSTM", "Logistic Regression", "ARIMA", "SVM"],
    "evaluation_metric": ["RMSE", "MAE", "F1 score", "accuracy", "ROC AUC"],
    "dataset": ["the Kaggle tourism dataset", "the UCI retail data", "the hotel bookings data"],
    "target_variable": ["tourist arrivals", "customer churn", "hotel occupancy"],
    "task_type": ["classification", "regression", "forecasting"],
}
PROPOSAL_TEMPLATES = ["Let's use {v}.", "I suggest {v}.", "What about {v}?", "We could go with {v}.",
                      "How about {v} for the {f}?", "I think {v} is the best option for the {f}."]
FIELD_WORDS = {"algorithm": "model", "evaluation_metric": "metric", "dataset": "dataset",
               "target_variable": "target", "task_type": "task"}
GENERIC = {"agreement": ["Agreed.", "Sounds good.", "Yes.", "Makes sense.", "Fine by me.", "Ok."],
           "rejection": ["No, I don't think so.", "Not sure that works.", "I disagree."],
           "finalization": ["Ok, let's lock it in.", "Decided then.", "Let's go with that."]}
SPECIFIC = {"agreement": ["Yes, {v} works.", "Agreed on {v}.", "{v} sounds good."],
            "rejection": ["No, not {v}.", "I don't think {v} fits."],
            "finalization": ["Let's finalize {v}.", "{v} it is."]}
FIELD_SPECIFIC = {"agreement": ["For the {f}, agreed.", "The {f} is fine."],
                  "rejection": ["I don't like that {f}."], "finalization": ["Let's lock in the {f}."]}
SMALL_TALK = ["What time is the meeting?", "I'll push my code tonight.", "lol", "brb", "Did anyone see the email?"]

ASSUMPTIONS = {
    "reply_to_latest": 0.65,          # a reply answers the newest open proposal
    "names_value": 0.30,              # ... and names the value
    "names_field": 0.20,              # ... or names the field instead
    "small_talk_rate": 0.15,
    "seconds_between": (5, 240),
}


@dataclass
class Chat:
    messages: list[Message] = dc_field(default_factory=list)
    parent: dict[str, str] = dc_field(default_factory=dict)       # reply id -> proposal id
    intent: dict[str, str] = dc_field(default_factory=dict)


def simulate_chat(rng, n_messages: int = 40, a: dict = ASSUMPTIONS) -> Chat:
    speakers = [f"M-{i}" for i in range(1, int(rng.integers(2, 6)) + 1)]
    chat, clock, open_props = Chat(), 0.0, []
    for i in range(n_messages):
        clock += float(rng.uniform(*a["seconds_between"]))
        mid = f"x{i}"
        r = rng.random()
        if not open_props or r < 0.3:                                  # a new proposal
            field = str(rng.choice(list(VALUES)))
            value = str(rng.choice(VALUES[field]))
            text = str(rng.choice(PROPOSAL_TEMPLATES)).format(v=value, f=FIELD_WORDS[field])
            msg = Message(mid, str(rng.choice(speakers)), clock, text, field, value)
            chat.messages.append(msg)
            open_props = [p for p in open_props if p.field != field][-4:] + [msg]
        elif r < 0.3 + a["small_talk_rate"]:
            chat.messages.append(Message(mid, str(rng.choice(speakers)), clock, str(rng.choice(SMALL_TALK))))
        else:                                                          # a reply
            if rng.random() < a["reply_to_latest"] or len(open_props) == 1:
                target = open_props[-1]
            else:
                target = open_props[int(rng.integers(0, len(open_props) - 1))]
            others = [s for s in speakers if s != target.speaker] or speakers
            who = str(rng.choice(others))
            intent = str(rng.choice(["agreement", "agreement", "agreement", "rejection", "finalization"]))
            u = rng.random()
            if u < a["names_value"]:
                text = str(rng.choice(SPECIFIC[intent])).format(v=target.value)
            elif u < a["names_value"] + a["names_field"]:
                text = str(rng.choice(FIELD_SPECIFIC[intent])).format(f=FIELD_WORDS[target.field])
            else:
                text = str(rng.choice(GENERIC[intent]))
            chat.messages.append(Message(mid, who, clock, text))
            chat.parent[mid] = target.message_id
            chat.intent[mid] = intent
    return chat


def generate_chats(n: int, seed: int = 5, a: dict = ASSUMPTIONS) -> list[Chat]:
    rng = np.random.default_rng(seed)
    return [simulate_chat(rng, int(rng.integers(20, 60)), a) for _ in range(n)]
