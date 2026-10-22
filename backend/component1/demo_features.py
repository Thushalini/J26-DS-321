"""Try every Component 1 feature on SAMPLE chats and see what Buddy says and what the blueprint holds.

Run from the backend folder:
    python -m component1.demo_features                 (every scenario)
    python -m component1.demo_features domino plan     (only the ones named)
    python -m component1.demo_features --list          (the scenario names)
    python -m component1.demo_features --no-nli        (compare known names only, no NLI model)

Rules-only extraction: nothing is sent to Groq or Gemini. The state, intent, reply-linking and
feasibility models are loaded the way the app loads them. Time is simulated: one message every
20 seconds. The same messages typed into the app should give the same replies, except that the
app also asks Groq/Gemini about messages the rules cannot read.
"""
from __future__ import annotations

import argparse
import os

SECONDS_BETWEEN_MESSAGES = 20

# name -> (what it shows, messages as (speaker, text) or (speaker, text, reply_to), quiet seconds after)
SCENARIOS = {
    "extract": ("extraction, uncertain values and a gap question", [
        ("M-1", "Our project is in the tourism domain."),
        ("M-2", "We want to predict monthly tourist arrivals to Sri Lanka."),
        ("M-1", "It's a regression problem, maybe we use random forest."),
    ], (110, 250)),
    "conflict": ("two different values: a conflict", [
        ("M-1", "Let's use random forest for the model."),
        ("M-2", "I think we should use XGBoost instead."),
    ], ()),
    "abbrev": ("an abbreviation is the same choice, not a conflict", [
        ("M-1", "Let's use RF for the model."),
        ("M-2", "Yes, random forest is a good choice."),
    ], ()),
    "meaning-diff": ("different meaning (NLI): a conflict", [
        ("M-1", "The target variable is tourist arrivals."),
        ("M-2", "The target variable is hotel occupancy."),
    ], ()),
    "meaning-same": ("same meaning (NLI): agreement", [
        ("M-1", "The target variable is monthly tourist arrivals."),
        ("M-2", "The target variable is tourist arrivals."),
    ], ()),
    "decide": ("a decision: proposed with a reason, agreed, finalized", [
        ("M-1", "Let's use F1 score as the metric because the classes are imbalanced."),
        ("M-2", "Agreed."),
        ("M-1", "Great, let's lock that in."),
    ], ()),
    "reject": ("a proposal rejected", [
        ("M-1", "Let's use accuracy as the metric."),
        ("M-2", "No, I don't think accuracy works here."),
    ], ()),
    "link": ("reply linking: 'agreed' goes to the proposal it is about, not the newest", [
        ("M-1", "Let's use random forest for the model."),
        ("M-2", "We could use F1 score as the metric."),
        ("M-3", "For the model, agreed."),
    ], ()),
    "link-button": ("reply linking with the app's Reply button", [
        ("M-1", "Let's use random forest for the model."),
        ("M-2", "We could use F1 score as the metric."),
        ("M-3", "Agreed.", "m1"),
    ], ()),
    "domino": ("Domino validation: changing the task re-checks the finalized metric", [
        ("M-1", "This is a classification task."),
        ("M-2", "Agreed, let's lock that in."),
        ("M-1", "We will use accuracy as the metric."),
        ("M-2", "Agreed, finalize it."),
        ("M-3", "Actually it should be a regression task, we predict the number of arrivals."),
        ("M-1", "Agreed, lock it in."),
    ], ()),
    "plan": ("plan checks and the feasibility model", [
        ("M-1", "It's a regression task."),
        ("M-2", "We'll use deep learning with a neural network."),
        ("M-1", "The dataset has 500 rows."),
        ("M-3", "We only have a laptop, no GPU."),
        ("M-2", "The deadline is in 2 weeks."),
        ("M-1", "Let's use accuracy as the metric."),
        ("M-3", "We can use Google Colab for the GPU."),
        ("M-1", "Let's use RMSE as the metric instead."),
    ], ()),
}

SAMPLE_DOCUMENT = """Project Brief: Sri Lanka Tourism Forecasting

This project is in the tourism domain. We aim to forecast monthly tourist arrivals to Sri Lanka.
The dataset is the SLTDA monthly arrivals dataset. The dataset has 3,000 rows.
We will evaluate the model using RMSE. Training will run on Google Colab.
The final report deadline is 15 December 2026.
"""


def show_buddy(replies: list[str]) -> None:
    for reply in replies:
        lines = reply.split("\n")
        print(f"    Buddy: {lines[0]}")
        for line in lines[1:]:
            print(f"           {line}")


def show_blueprint(agent, with_plan: bool) -> None:
    from .decisions.view import decisions_view
    from .plan import plan_from_events
    from .timeline.current import current_timeline

    state = agent.store.state
    print("  Fields:")
    for name, f in state.fields.items():
        if f.status.value != "missing":
            print(f"    {name.value:18s} {f.status.value:12s} {f.value}")
    view = decisions_view(state, agent.store.events)
    if view["all"]:
        print("  Decisions:")
    for d in view["all"]:
        extra = [f"agreed by {', '.join(d['agreed_by'])}" if d.get("agreed_by") else "",
                 f"objected by {', '.join(d['objected_by'])}" if d.get("objected_by") else "",
                 f"why: {d['rationale']}" if d.get("rationale") else "",
                 f"re-check: {d['pending_reason']}" if d.get("pending_reason") else ""]
        print(f"    {d['field']:18s} {d['status']:12s} {d['value']!r} (proposed by {d['proposed_by']})"
              + "".join(f"; {e}" for e in extra if e))
    if not with_plan:
        return
    plan = plan_from_events(agent.store.project_id, list(agent.store.events))
    print("  Plan checks:")
    for c in plan.get("checks") or []:
        print(f"    {c['status']:8s} {c['label']}" + (f": {c['detail']}" if c.get("detail") else ""))
    model = plan.get("feasibility_model")
    if model:
        print(f"  Feasibility: {model['label']} ({model['probability']:.0%})")
        for r in model.get("reasons") or []:
            print(f"    reason: {r['text']} ({r['effect']})")
        for a in model.get("alternatives") or []:
            print(f"    change: {a['change']} -> {a['label']} ({a['feasible_probability']:.0%} feasible)")
    t = current_timeline(state, agent.store.events)
    if t:
        print(f"  Timeline: {t['text']} -> {t.get('deadline')} ({t.get('days_left')} days left)")


def run_scenario(name: str, bundle, llm, intent_bundle) -> None:
    from .agent.chat_agent import ChatAgent

    about, messages, quiet = SCENARIOS[name]
    print(f"\n=== {name}: {about} ===")
    agent = ChatAgent(f"demo-{name}", bundle, llm, intent_bundle=intent_bundle)
    now = 0.0
    for i, m in enumerate(messages):
        now = i * SECONDS_BETWEEN_MESSAGES
        reply_to = m[2] if len(m) > 2 else None
        print(f"[m{i + 1}] {m[0]}: {m[1]}" + (f"   (Reply button on m{reply_to[1:]})" if reply_to else ""))
        show_buddy(agent.on_message(f"m{i + 1}", m[0], m[1], now, reply_to=reply_to))
    for wait in quiet:
        print(f"   ... {wait} seconds of quiet ...")
        replies = agent.tick(now + wait)
        show_buddy(replies) if replies else print("    (Buddy stays silent)")
    show_blueprint(agent, with_plan=(name == "plan"))


def run_document() -> None:
    from .documents.chunk import chunk_pages
    from .documents.extract import extract_document
    from .documents.parse import parse_document

    print("\n=== document: suggestions found in an uploaded file (rules only) ===")
    pages = parse_document("project_brief.txt", SAMPLE_DOCUMENT.encode("utf-8"))
    for fact in extract_document(chunk_pages(pages), fallback_llm=None):
        print(f"    {fact.field:18s} {fact.value!r}   from: \"{fact.span}\"")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="scenarios to run (default: all, plus 'document')")
    ap.add_argument("--list", action="store_true", help="list the scenario names and stop")
    ap.add_argument("--no-nli", action="store_true", help="switch the NLI model off")
    args = ap.parse_args(argv)
    if args.list:
        for name, (about, _, _) in SCENARIOS.items():
            print(f"{name:13s} {about}")
        print(f"{'document':13s} suggestions found in an uploaded file")
        return 0
    os.environ["C1_EXTRACTOR_MODE"] = "rules"
    os.environ["C1_NLI"] = "off" if args.no_nli else "on"

    from .extraction.llm import build_extractor_llm
    from .intent.predict import MODEL_PATH as INTENT_PATH, load_model as load_intent
    from .state.predict import MODEL_PATH, load_model

    if not MODEL_PATH.exists():
        print("The state model is missing. Run first:  python -m component1.state.train")
        return 1
    names = args.names or list(SCENARIOS) + ["document"]
    unknown = [n for n in names if n not in SCENARIOS and n != "document"]
    if unknown:
        print(f"Unknown scenario(s): {', '.join(unknown)}. See --list.")
        return 1
    bundle, (llm, _) = load_model(), build_extractor_llm("rules")
    intent_bundle = load_intent(INTENT_PATH) if INTENT_PATH.exists() else None
    print("SAMPLE MESSAGES ONLY. Rules-only extraction: nothing is sent anywhere."
          + ("" if intent_bundle else "\n(The intent model is missing, so keyword rules decide agreement.)"))
    for name in names:
        run_document() if name == "document" else run_scenario(name, bundle, llm, intent_bundle)
    print("\nAll models were trained on synthetic or simulated data; their confidences are not calibrated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
