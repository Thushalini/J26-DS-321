"""The hand-written baseline for decision intent: the word clues from the labeling guide.

A trained model must beat this to be worth having. The priority order is the guide's:
finalization > rejection > proposal > agreement > other.
"""
from __future__ import annotations

CLOSING = ("finalize", "finalise", "decided", "locked in", "lock it in", "we'll go with",
           "we will go with", "let's go with", "lets go with", "settled")
REFUSING = ("i don't think", "i dont think", "let's not", "lets not", "scrap that",
            "not a good idea", "i disagree", "no,", "no.", "nope")
OFFERING = ("i suggest", "what about", "we could", "how about", "maybe we", "why don't we",
            "why dont we", "let's use", "lets use", "let's try", "lets try", "we should", "i think we should")
ENDORSING = ("i agree", "agreed", "sounds good", "makes sense", "fine by me", "good idea",
             "yes", "yeah", "yep", "sure", "okay", "ok")


def keyword_intent(message: str, prev_message: str = "") -> str:
    text = " " + message.lower().strip() + " "
    for words, label in ((CLOSING, "finalization"), (REFUSING, "rejection"), (OFFERING, "proposal")):
        if any(w in text for w in words):
            return label
    if prev_message and any(f" {w}" in text or text.strip().startswith(w) for w in ENDORSING):
        return "agreement"
    return "other"