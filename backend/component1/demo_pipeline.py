"""Run five SAMPLE messages through the whole connected pipeline.

Run from the backend folder:
    python -m component1.demo_pipeline --fake         (pretend model, nothing sent anywhere)
    python -m component1.demo_pipeline                (as the app is set up: rules, then Groq/Gemini)
    python -m component1.demo_pipeline --mode rules   (rules only: no key, nothing sent)
    python -m component1.demo_pipeline --mode llm     (the language model for every message)
    python -m component1.demo_pipeline --delay 30     (wait between messages, for rate limits)

Train the state model first:  python -m component1.state.train
SAMPLE MESSAGES ONLY. On a free tier the provider (Groq or Google) may keep what you send.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .agent.signals import CONFLICT, GAP
from .common.events import EventStore
from .common.schema import FieldName
from .extraction.demo_extract import SAMPLE_MESSAGES, add_extractor_options, setup_extractor
from .gaps.gaps import check_completeness
from .pipeline import process_message
from .state.predict import MODEL_PATH, load_model

# What a good model would return for each sample message. Used by --fake.
SCRIPT = {
    "I think we should predict tourist arrivals.": [
        {"field": "target_variable", "value": "tourist arrivals", "span": "predict tourist arrivals", "hedged": True}],
    "Let's use XGBoost, it handles tabular data well.": [
        {"field": "algorithm", "value": "XGBoost", "span": "use XGBoost", "hedged": False}],
    "What time is the meeting tomorrow?": [],
    "Maybe RMSE for the metric? Not sure.": [
        {"field": "evaluation_metric", "value": "RMSE", "span": "RMSE", "hedged": True}],
    "Actually the target should be hotel occupancy.": [
        {"field": "target_variable", "value": "hotel occupancy", "span": "hotel occupancy", "hedged": False}],
}


def scripted_llm(prompt: str) -> str:
    for text, items in SCRIPT.items():
        if text in prompt:
            return json.dumps({"items": items})
    return json.dumps({"items": []})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="use a pretend model (scripted answers), send nothing")
    add_extractor_options(ap)
    args = ap.parse_args(argv)

    if not MODEL_PATH.exists():
        print("The state model is missing. Run first:  python -m component1.state.train")
        return 1
    bundle = load_model()

    if args.fake:
        llm = scripted_llm
        print("PRETEND MODEL (scripted answers). Nothing is sent anywhere.\n")
    else:
        llm = setup_extractor(args.mode)

    store = EventStore("demo")
    for i, (message_id, speaker, text) in enumerate(SAMPLE_MESSAGES):
        if i and args.delay and getattr(llm, "mode", "rules") != "rules":
            time.sleep(args.delay)
        print(f"{speaker}: {text}")
        try:
            result = process_message(store, bundle, llm, message_id, speaker, text)
        except Exception as exc:
            print(f"\nFailed: {type(exc).__name__}: {exc}")
            return 1
        if not result.saved:
            print("   -> nothing saved")
        for field, value in result.saved:
            label, probability = result.statuses[field]
            print(f"   -> saved {field} = {value}   status: {label} (model confidence {probability:.2f})")
        for signal in result.signals:
            if signal.kind == CONFLICT:
                print(f"   !! conflict on {signal.field}: {' / '.join(signal.values)}")
        gaps = [s.field for s in result.signals if s.kind == GAP]
        if gaps:
            print(f"   next gap to ask about: {gaps[0]}")

    print("\nBlueprint after five messages:")
    state = store.state
    for name in FieldName:
        f = state.fields[name]
        value = f.value if f.value is not None else "-"
        conf = f"{f.confidence:.2f}" if f.value is not None else "-"
        print(f"  {name.value:<18} {value:<22} {f.status.value:<12} {conf}")

    report = check_completeness(state)
    print(f"\nPlan defined enough to start building? {report.complete}")
    print("Model confidence is NOT calibrated, and the state model was trained on synthetic data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
