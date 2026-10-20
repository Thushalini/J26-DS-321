"""A chat transcript: five SAMPLE messages and Buddy's replies.

Run from the backend folder:
    python -m component1.demo_chat --fake        (pretend model, no key needed)
    python -m component1.demo_chat               (real Gemini, sample messages only)
    python -m component1.demo_chat --delay 30    (wait between Gemini calls)

Train the state model first:  python -m component1.state.train
SAMPLE MESSAGES ONLY. On Gemini's free tier Google may keep what you send.

Time is simulated: each message arrives 20 seconds after the last, then the chat goes
quiet so you can see Buddy ask about a missing field.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .agent.chat_agent import ChatAgent
from .demo_pipeline import scripted_llm
from .extraction.demo_extract import SAMPLE_MESSAGES
from .extraction.gemini_provider import GeminiConfigError, load_env_file, make_gemini_llm
from .state.predict import MODEL_PATH, load_model

BACKEND_DIR = Path(__file__).resolve().parents[1]
SECONDS_BETWEEN_MESSAGES = 20
QUIET_TICKS = (110, 250, 400)   # seconds after the last message


def show_buddy(replies: list[str]) -> None:
    for reply in replies:
        lines = reply.split("\n")
        print(f"Buddy:    {lines[0]}")
        for line in lines[1:]:
            print(f"          {line}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="use a pretend model instead of Gemini")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between Gemini calls")
    args = ap.parse_args(argv)

    if not MODEL_PATH.exists():
        print("The state model is missing. Run first:  python -m component1.state.train")
        return 1
    bundle = load_model()

    if args.fake:
        llm = scripted_llm
        print("PRETEND MODEL (scripted answers). Nothing is sent anywhere.\n")
    else:
        load_env_file(BACKEND_DIR / ".env")
        try:
            llm = make_gemini_llm()
        except GeminiConfigError as exc:
            print(f"Cannot start: {exc}")
            return 1
        print("SAMPLE MESSAGES ONLY. Nothing real is being sent.\n")

    agent = ChatAgent("demo", bundle, llm)
    now = 0.0
    for i, (message_id, speaker, text) in enumerate(SAMPLE_MESSAGES):
        if i and args.delay and not args.fake:
            time.sleep(args.delay)
        now = i * SECONDS_BETWEEN_MESSAGES
        print(f"{speaker}: {text}")
        try:
            show_buddy(agent.on_message(message_id, speaker, text, now))
        except Exception as exc:
            print(f"\nFailed: {type(exc).__name__}: {exc}")
            return 1

    last_message_at = now
    for wait in QUIET_TICKS:
        print(f"\n   ... {wait} seconds of quiet ...")
        replies = agent.tick(last_message_at + wait)
        show_buddy(replies)
        if not replies:
            print("   (Buddy stays silent)")
    print("\nModel confidence is NOT calibrated, and the state model was trained on synthetic data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())