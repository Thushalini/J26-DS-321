"""Run extraction on five SAMPLE messages and show what gets saved.

It extracts the same way the app does: rules first, then the language model chosen in
backend/.env (Groq or Gemini, see C1_LLM_PROVIDER) only when the rules find nothing.

Run from the backend folder:
    python -m component1.extraction.demo_extract                 (as the app is set up)
    python -m component1.extraction.demo_extract --mode rules    (rules only: no key, nothing sent)
    python -m component1.extraction.demo_extract --mode llm      (the language model for every message)
    python -m component1.extraction.demo_extract --show-prompt   (print the prompt and stop)
    python -m component1.extraction.demo_extract --delay 30      (wait between messages, for rate limits)

SAMPLE MESSAGES ONLY. They are invented. On a free tier the provider (Groq or Google) may
keep and review what you send, so never put real people's chat here without consent.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..common.events import EventStore
from ..common.schema import FieldName
from .extractor import apply_message, build_prompt
from .gemini_provider import load_env_file
from .hybrid import MODES
from .llm import build_extractor_llm

SAMPLE_MESSAGES = [
    ("M0001", "Member_A", "I think we should predict tourist arrivals."),
    ("M0002", "Member_B", "Let's use XGBoost, it handles tabular data well."),
    ("M0003", "Member_C", "What time is the meeting tomorrow?"),
    ("M0004", "Member_A", "Maybe RMSE for the metric? Not sure."),
    ("M0005", "Member_C", "Actually the target should be hotel occupancy."),
]

BACKEND_DIR = Path(__file__).resolve().parents[2]


def add_extractor_options(ap: argparse.ArgumentParser) -> None:
    """The options every demo shares."""
    ap.add_argument("--mode", choices=MODES, default=None,
                    help="hybrid / rules / llm (default: C1_EXTRACTOR_MODE in .env, else hybrid)")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to wait between messages (free-tier rate limits)")


def setup_extractor(mode):
    """Load backend/.env and build the extractor. Prints how it extracts."""
    load_env_file(BACKEND_DIR / ".env")
    llm, how = build_extractor_llm(mode)
    print(f"Extracting with: {how}")
    if llm.mode != "rules":
        print("SAMPLE MESSAGES ONLY. Messages the rules cannot read are sent to the provider.")
    print()
    return llm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-prompt", action="store_true", help="print the prompt for the first message and stop")
    add_extractor_options(ap)
    args = ap.parse_args(argv)

    if args.show_prompt:
        print(build_prompt(SAMPLE_MESSAGES[0][2]))
        return 0

    llm = setup_extractor(args.mode)
    store = EventStore("demo")
    for i, (message_id, speaker, text) in enumerate(SAMPLE_MESSAGES):
        if i and args.delay and llm.mode != "rules":
            time.sleep(args.delay)
        print(f"{speaker}: {text}")
        try:
            events = apply_message(store, message_id, speaker, text, llm)
        except Exception as exc:  # show the problem plainly, without any secrets
            print(f"\nThe language model call failed: {type(exc).__name__}: {exc}")
            print("Check: the key, the model name, your internet, or a rate limit (try --delay 30).")
            return 1
        if not events:
            print("   -> nothing saved")
        for event in events:
            p = event.payload
            print(f"   -> saved {p['field']} = {p['new_value']}  ({p['new_status']})")

    print("\nBlueprint after five messages:")
    state = store.state
    for name in FieldName:
        f = state.fields[name]
        value = f.value if f.value is not None else "-"
        print(f"  {name.value:<18} {value:<22} {f.status.value}")
    print("\nThis demo only extracts. Conflicts and statuses: python -m component1.demo_pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
