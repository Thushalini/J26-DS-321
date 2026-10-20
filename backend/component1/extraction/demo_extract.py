"""Run extraction on five SAMPLE messages with Gemini and show what gets saved.

Run from the backend folder:
    python -m component1.extraction.demo_extract                 (calls Gemini)
    python -m component1.extraction.demo_extract --show-prompt   (no key needed)
    python -m component1.extraction.demo_extract --delay 30      (wait between calls)

SAMPLE MESSAGES ONLY. They are invented. On Gemini's free tier, Google may keep and
review what you send, so never put real people's chat here unless you are on the
paid tier and have consent.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..common.events import EventStore
from ..common.schema import FieldName
from .extractor import apply_message, build_prompt
from .gemini_provider import GeminiConfigError, load_env_file, make_gemini_llm

SAMPLE_MESSAGES = [
    ("M0001", "Member_A", "I think we should predict tourist arrivals."),
    ("M0002", "Member_B", "Let's use XGBoost, it handles tabular data well."),
    ("M0003", "Member_C", "What time is the meeting tomorrow?"),
    ("M0004", "Member_A", "Maybe RMSE for the metric? Not sure."),
    ("M0005", "Member_C", "Actually the target should be hotel occupancy."),
]

BACKEND_DIR = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-prompt", action="store_true", help="print the prompt for the first message and stop")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between calls (free-tier rate limits)")
    args = ap.parse_args(argv)

    if args.show_prompt:
        print(build_prompt(SAMPLE_MESSAGES[0][2]))
        return 0

    load_env_file(BACKEND_DIR / ".env")
    try:
        llm = make_gemini_llm()
    except GeminiConfigError as exc:
        print(f"Cannot start: {exc}")
        return 1

    print("SAMPLE MESSAGES ONLY. Nothing real is being sent.\n")
    store = EventStore("demo")
    for i, (message_id, speaker, text) in enumerate(SAMPLE_MESSAGES):
        if i and args.delay:
            time.sleep(args.delay)
        print(f"{speaker}: {text}")
        try:
            events = apply_message(store, message_id, speaker, text, llm)
        except Exception as exc:  # show the problem plainly, without any secrets
            print(f"\nThe Gemini call failed: {type(exc).__name__}: {exc}")
            print("Check: the key, the model name, your internet, billing/region, or a rate limit (try --delay 30).")
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
    print("\nKnown limit: there is no conflict detection yet, so message 5 simply replaces the target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())