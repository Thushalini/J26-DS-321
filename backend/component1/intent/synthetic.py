"""Create labeled training conversations with a language model, and check them with people.

Run from the backend folder:
    python -m component1.intent.synthetic --n 60 --out component1/data/intent_synthetic.csv
    python -m component1.intent.synthetic --review component1/data/intent_synthetic.csv --count 60

HOW IT WORKS: the model writes a short team chat about planning a data science project and
labels every turn, so the label is known by construction and nobody has to label by hand.
It uses the same language model as the rest of Component 1 (Groq or Gemini, set by C1_LLM_PROVIDER).

WHAT THAT DOES AND DOES NOT GIVE YOU:
  * Good for TRAINING data, especially for rare labels such as "finalization".
  * The model's labels can be wrong. Check a sample with people (--review) and report the agreement.
  * Models that play students tend to be too polite and too on-topic. Never use this data as
    the final test. The test must be messages written and labeled by people.
  * The provider may keep what you send. These prompts contain no real chat.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from .data import LABELS

LLMFn = Callable[[str], str]

DEFINITIONS = """Labels (use exactly these words):
- proposal: puts forward a choice for the project that the team has not settled (dataset, target, algorithm, metric, resources, plan). A question that names a specific option also counts as a proposal.
- agreement: supports a choice already put forward, without closing the decision.
- finalization: closes the decision. The team treats it as settled and moves on.
- rejection: says no to an option or withdraws support for it.
- other: anything else: facts, explanations, questions that only ask for information, scheduling, status updates, small talk."""

STYLES = [
    "short, casual chat messages with a few typos and no capital letters",
    "polite and fairly formal, full sentences",
    "very terse, many one-word or two-word replies",
    "friendly, with some jokes and off-topic remarks mixed in",
    "uncertain and hedging a lot ('maybe', 'not sure', 'I think')",
    "one person dominates and the others reply briefly",
    "messages that sometimes ask questions back before deciding",
    "a mix of long explanations and quick agreements",
]

TOPICS = [
    "predicting monthly tourist arrivals", "predicting customer churn for a telecom company",
    "forecasting electricity demand", "classifying product reviews by sentiment",
    "predicting house prices", "detecting fraudulent card transactions",
    "predicting student exam results", "forecasting rice yield from weather data",
    "predicting hospital bed occupancy", "recommending books to library members",
]

DECISIONS = ["the dataset", "the target variable", "the algorithm", "the evaluation metric", "the compute or time budget"]

# Cycled so every label, including the rare ones, appears often.
MUST_INCLUDE = [
    ["finalization", "agreement"], ["rejection", "proposal"], ["finalization", "rejection"],
    ["agreement", "other"], ["proposal", "finalization"], ["rejection", "agreement"],
]


def build_prompt(topic: str, style: str, decision: str, n_turns: int, speakers: int, must_include: list[str]) -> str:
    names = ", ".join("ABCD"[:speakers])
    return f"""Write a realistic group chat between {speakers} university students ({names}) planning a data science project about {topic}.
They discuss {decision}. Style: {style}.
Write exactly {n_turns} chat messages. Each message is one turn by one person.

{DEFINITIONS}

Label every message. Across the chat include at least one message for each of these labels: {", ".join(must_include)}.
Messages must read like real chat. Do not explain anything outside the chat.
Vary the wording a lot. Real students rarely use stock phrases, so do NOT use these: 'let's finalize', 'we're going with', 'sounds good', 'I agree', 'I suggest', 'what about', 'how about'. Express the same intents in other natural ways, for example 'works for me', 'ok do it', 'fine, that's the plan then', 'nah that won't work', 'wait, why not X instead'.
Include a few messages that are hard to label, for example a rejection of one option that also supports another.

Reply with JSON only, in this shape:
{{"turns": [{{"speaker": "A", "text": "ok lets try xgboost", "intent": "proposal"}}]}}"""


def _json_object(raw: str) -> Optional[dict]:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_conversation(raw: str, min_turns: int = 4, max_turns: int = 20) -> Optional[list[dict]]:
    """Return clean turns, or None if the reply is not a usable conversation."""
    data = _json_object(raw)
    turns = data.get("turns") if data else None
    if not isinstance(turns, list) or not (min_turns <= len(turns) <= max_turns):
        return None
    clean = []
    for t in turns:
        if not isinstance(t, dict):
            return None
        text = str(t.get("text", "")).strip()
        intent = str(t.get("intent", "")).strip().lower()
        speaker = str(t.get("speaker", "")).strip() or "?"
        if not text or len(text) > 400 or intent not in LABELS:
            return None
        clean.append({"speaker": speaker, "message": text, "intent": intent})
    return clean


def generate(llm: LLMFn, n: int, seed: int = 0, delay: float = 0.0, log: Callable[[str], None] = print) -> list[dict]:
    """Ask the model for n conversations. Unusable replies are skipped, so fewer may come back."""
    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()
    for i in range(n):
        if i and delay:
            time.sleep(delay)
        prompt = build_prompt(
            topic=rng.choice(TOPICS), style=STYLES[i % len(STYLES)], decision=rng.choice(DECISIONS),
            n_turns=rng.randint(6, 10), speakers=rng.randint(2, 4), must_include=MUST_INCLUDE[i % len(MUST_INCLUDE)],
        )
        try:
            turns = parse_conversation(llm(prompt))
        except Exception as exc:
            log(f"conversation {i + 1}: the model call failed ({type(exc).__name__}); skipped")
            continue
        if turns is None:
            log(f"conversation {i + 1}: unusable reply; skipped")
            continue
        fingerprint = " | ".join(t["message"].lower() for t in turns)
        if fingerprint in seen:
            log(f"conversation {i + 1}: duplicate; skipped")
            continue
        seen.add(fingerprint)
        conv_id = f"syn_{len(seen):04d}"
        for turn_no, t in enumerate(turns, start=1):
            rows.append({"conv_id": conv_id, "turn": turn_no, "speaker": t["speaker"],
                         "message": t["message"], "intent": t["intent"], "source": "generated"})
    return rows


def write_csv(rows: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["conv_id", "turn", "speaker", "message", "intent", "source"])
        w.writeheader()
        w.writerows(rows)


def make_review_sheet(csv_in, sheet_out, key_out, count: int = 60, seed: int = 1) -> int:
    """Pick a random sample of generated messages for a PERSON to label without seeing the model's label.

    sheet_out: msg_id, speaker, prev_message, message, intent (blank, for the person to fill in).
    key_out:   msg_id, intent (what the model said). Compare them with:
               python -m component1.labeling.kappa <filled sheet> <key>
    """
    with open(csv_in, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_conv: dict[str, list[dict]] = {}
    for r in rows:
        by_conv.setdefault(r["conv_id"], []).append(r)
    items = []
    for conv_id, turns in by_conv.items():
        turns.sort(key=lambda r: int(r["turn"]))
        prev = ""
        for r in turns:
            items.append({"msg_id": f"{conv_id}:{r['turn']}", "speaker": r["speaker"],
                          "prev_message": prev, "message": r["message"], "key": r["intent"]})
            prev = r["message"]
    random.Random(seed).shuffle(items)
    sample = items[:count]
    with open(sheet_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["msg_id", "speaker", "prev_message", "message", "intent"])
        w.writeheader()
        for s in sample:
            w.writerow({"msg_id": s["msg_id"], "speaker": s["speaker"], "prev_message": s["prev_message"],
                        "message": s["message"], "intent": ""})
    with open(key_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["msg_id", "intent"])
        w.writeheader()
        for s in sample:
            w.writerow({"msg_id": s["msg_id"], "intent": s["key"]})
    return len(sample)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0, help="how many conversations to generate")
    ap.add_argument("--out", default="component1/data/intent_synthetic.csv")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between calls (rate limits)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--review", help="make a review sheet from this generated file instead of generating")
    ap.add_argument("--count", type=int, default=60, help="messages in the review sheet")
    args = ap.parse_args(argv)

    if args.review:
        base = Path(args.review)
        sheet, key = base.with_name(base.stem + "_review_sheet.csv"), base.with_name(base.stem + "_review_key.csv")
        n = make_review_sheet(args.review, sheet, key, args.count)
        print(f"Wrote {n} messages to {sheet} (give this to a person) and the model's labels to {key} (keep it hidden).")
        return 0
    if args.n <= 0:
        ap.error("give --n (how many conversations) or --review")

    from ..extraction.gemini_provider import load_env_file
    from ..extraction.llm import make_configured_llm
    load_env_file(Path(__file__).resolve().parents[2] / ".env")
    try:
        llm = make_configured_llm()      # Groq or Gemini, as set by C1_LLM_PROVIDER
    except Exception as exc:
        print(f"Cannot start: {type(exc).__name__}: {exc}")
        return 1
    rows = generate(llm, args.n, seed=args.seed, delay=args.delay)
    write_csv(rows, args.out)
    counts = {l: sum(r["intent"] == l for r in rows) for l in LABELS}
    print(f"Wrote {len(rows)} messages in {len({r['conv_id'] for r in rows})} conversations to {args.out}")
    print("Label counts:", counts)
    print("Next: check a sample with a person (--review), then train: python -m component1.intent.train", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())