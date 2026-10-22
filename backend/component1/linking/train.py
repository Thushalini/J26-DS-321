"""Train the reply linker and compare it with the "latest proposal" baseline.

Run from the backend folder:
    python -m component1.linking.train

Each reply becomes a set of (reply, candidate) pairs, one per recent proposal; the true parent is
the positive. A classifier scores pairs; for each reply the highest-scoring candidate is chosen.
It is judged by how often that choice is the true parent (top-1 accuracy), on chats it never saw,
split by whole chat. Also reported separately: replies that do NOT answer the newest proposal,
where the baseline is always wrong by construction.

READ BEFORE QUOTING ANY NUMBER: the chats are SIMULATED (simulate.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES
from .predict import pairs
from .simulate import ASSUMPTIONS, generate_chats

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "reply_linker.joblib"
RESULTS_PATH = HERE / "results" / "reply_linker_results.json"


def build(chats):
    """X, y, and per reply: (row indexes, index of the true parent among them, is_latest_true)."""
    X, y, groups = [], [], []
    for chat in chats:
        history, answered = [], {}
        for m in chat.messages:
            if m.message_id in chat.parent:
                found = pairs(m, history, answered)
                ids = [c.message_id for c, _ in found]
                true = chat.parent[m.message_id]
                if found and true in ids:
                    start = len(X)
                    for cand, f in found:
                        X.append(f)
                        y.append(1 if cand.message_id == true else 0)
                    groups.append((list(range(start, len(X))), ids.index(true), ids.index(true) == len(ids) - 1))
                answered.setdefault(true, set()).add(m.speaker)
            history.append(m)
    return np.array(X), np.array(y), groups


def top1(scores, groups) -> dict:
    hits = [int(np.argmax(scores[rows])) == true for rows, true, _ in groups]
    older = [h for h, (_, _, latest) in zip(hits, groups) if not latest]
    return {"top1_accuracy": round(float(np.mean(hits)), 3),
            "top1_on_replies_to_older_proposals": round(float(np.mean(older)), 3) if older else None,
            "replies": len(groups)}


def baseline(groups) -> dict:
    hits = [true == len(rows) - 1 for rows, true, _ in groups]
    return {"top1_accuracy": round(float(np.mean(hits)), 3), "top1_on_replies_to_older_proposals": 0.0,
            "replies": len(groups)}


def train_and_evaluate(n_chats: int = 600, seed: int = 5, save: bool = True) -> dict:
    chats = generate_chats(n_chats, seed=seed)
    cut = int(len(chats) * 0.8)
    X_tr, y_tr, _ = build(chats[:cut])
    X_te, y_te, g_te = build(chats[cut:])

    forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, random_state=seed).fit(X_tr, y_tr)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
    results = {
        "data": {"kind": "SIMULATED planning chats (simulate.py). Not real chat.",
                 "chats_train": cut, "chats_test": len(chats) - cut, "pairs_train": int(len(y_tr)),
                 "replies_test": len(g_te),
                 "share_of_test_replies_to_older_proposals": round(float(np.mean([not l for _, _, l in g_te])), 3),
                 "assumptions": ASSUMPTIONS},
        "baseline_latest_proposal": baseline(g_te),
        "logistic_regression": top1(logistic.predict_proba(X_te)[:, 1], g_te),
        "random_forest": top1(forest.predict_proba(X_te)[:, 1], g_te),
        "random_forest_feature_importance": {
            n: round(float(v), 3) for n, v in sorted(zip(FEATURE_NAMES, forest.feature_importances_), key=lambda kv: -kv[1])
        },
    }
    best_name = max(("logistic_regression", "random_forest"), key=lambda k: results[k]["top1_accuracy"])
    best = forest if best_name == "random_forest" else logistic
    results["saved_model"] = best_name
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": best, "features": FEATURE_NAMES, "kind": best_name}, MODEL_PATH)
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> int:
    r = train_and_evaluate()
    print("SIMULATED DATA. These numbers do not prove the linker works on real chat.\n")
    print(f"Test replies: {r['data']['replies_test']}  (answering an OLDER proposal: "
          f"{r['data']['share_of_test_replies_to_older_proposals']:.0%})")
    for name in ("baseline_latest_proposal", "logistic_regression", "random_forest"):
        s = r[name]
        print(f"  {name:<26} top-1 {s['top1_accuracy']:.3f}   on replies to older proposals "
              f"{s['top1_on_replies_to_older_proposals']}")
    print("\nFeature importance:", r["random_forest_feature_importance"])
    print(f"Saved: {r['saved_model']} -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
