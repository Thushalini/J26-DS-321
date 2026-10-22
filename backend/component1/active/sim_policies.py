"""Does asking by priority help? Simulated teams planning with four question strategies.

    python -m component1.active.sim_policies

Every 6 messages the simulated Buddy asks one question and the team's next message answers it
(simulate.py: a gap gets a first answer, a confirm request either brings a hidden disagreement out
now or gets an agreement, a clarify request gets a specific value). A quarter of first answers are
too vague. Strategies:
    none       Buddy asks nothing
    fixed      the old order: the first missing field by importance weight (gaps.ready_gaps)
    priority   the new order: gap, confirm or clarify question with the highest priority (priority.py)
    random     a random question from the same choices as priority
Every strategy plays the same teams (same seeds). Measured per team:
    messages until the plan is complete (every field settled, specific and undisputed; teams that
    never get there count as the full length), rework (a value changed after fields that depend on it
    were already set), changes in total, and vague values left at the end.
Results go to results/policy_results.json. SIMULATED: the same assumptions made the risk model.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..agent.signals import CLARIFY, CONFIRM, GAP
from ..gaps.gaps import ready_gaps
from .priority import question_signals
from .risk import default_bundle
from .simulate import VALUES, simulate_team

RESULTS = Path(__file__).parent / "results" / "policy_results.json"
N_TEAMS = 150
VAGUE_RATE = 0.25
ASK_EVERY = 6
KIND = {GAP: "gap", CONFIRM: "confirm", CLARIFY: "clarify"}
MEASURES = ("messages_to_complete", "rework", "changes", "vague_left")


def fixed(state, events, days_left):
    gaps = [g.value for g in ready_gaps(state) if g.value in VALUES]
    return ("gap", gaps[0]) if gaps else None


def make_priority(bundle):
    def ask(state, events, days_left):
        for s in question_signals(state, events, bundle, days_left, with_reasons=False):
            if s.field in VALUES:
                return KIND[s.kind], s.field
        return None
    return ask


def make_random(bundle, rng):
    def ask(state, events, days_left):
        options = [(KIND[s.kind], s.field) for s in question_signals(state, events, bundle, days_left, with_reasons=False)
                   if s.field in VALUES]
        return options[int(rng.integers(len(options)))] if options else None
    return ask


def play(strategy: str, team_seed: int, bundle=None) -> dict:
    bundle = bundle if bundle is not None else default_bundle()
    rng = np.random.default_rng(team_seed)
    asker = {"none": None, "fixed": fixed, "priority": make_priority(bundle),
             "random": make_random(bundle, np.random.default_rng(team_seed + 10_000))}[strategy]
    out: dict = {}
    simulate_team(rng, asker=asker, vague_rate=VAGUE_RATE, ask_every=ASK_EVERY, outcome=out, snapshots=False)
    done = out["completed_at"]
    return {"messages_to_complete": done if done is not None else out["steps"], "completed": done is not None,
            "rework": out["rework"], "changes": out["changes"], "vague_left": out["vague_left"], "asked": out["asked"]}


def paired_ci(a: np.ndarray, b: np.ndarray, seed: int = 0, n: int = 2000) -> list[float]:
    """95% bootstrap interval of mean(a - b) over teams."""
    rng = np.random.default_rng(seed)
    d = a - b
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]


def main(n_teams: int = N_TEAMS, seed: int = 100) -> dict:
    bundle = default_bundle()
    if bundle is None:
        raise SystemExit("Train the risk model first: python -m component1.active.train_risk")
    runs = {}
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:      # each team is independent
        for strategy in ("none", "fixed", "priority", "random"):
            print(f"Playing {n_teams} teams with strategy '{strategy}' ({workers} processes)...")
            runs[strategy] = list(pool.map(play, [strategy] * n_teams, [seed + i for i in range(n_teams)]))

    results = {"data": {"kind": "SIMULATED teams (active/simulate.py, sim_policies.py). Not real chat.",
                        "teams": n_teams, "vague_rate": VAGUE_RATE, "ask_every": ASK_EVERY},
               "strategies": {}}
    for strategy, rows in runs.items():
        results["strategies"][strategy] = {
            **{m: round(float(np.mean([r[m] for r in rows])), 2) for m in MEASURES},
            "completed_share": round(float(np.mean([r["completed"] for r in rows])), 3),
            "questions_asked": round(float(np.mean([r["asked"] for r in rows])), 1),
        }
    for other in ("fixed", "random"):
        key = f"priority_minus_{other}"
        results[key] = {}
        for m in MEASURES:
            a = np.array([r[m] for r in runs["priority"]], dtype=float)
            b = np.array([r[m] for r in runs[other]], dtype=float)
            results[key][m] = {"mean": round(float((a - b).mean()), 2), "ci95": paired_ci(a, b)}

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSIMULATED TEAMS. These numbers do not prove the strategy works on real teams.\n")
    print(f"{'strategy':10s} {'msgs to complete':>17s} {'completed':>10s} {'rework':>7s} {'changes':>8s} {'vague left':>11s}")
    for s, r in results["strategies"].items():
        print(f"{s:10s} {r['messages_to_complete']:17.1f} {r['completed_share']:10.0%} {r['rework']:7.2f} "
              f"{r['changes']:8.2f} {r['vague_left']:11.2f}")
    for other in ("fixed", "random"):
        print(f"\npriority - {other} (mean difference per team, 95% interval):")
        for m, v in results[f"priority_minus_{other}"].items():
            print(f"  {m:22s} {v['mean']:+.2f}  [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
    return results


if __name__ == "__main__":
    main()
