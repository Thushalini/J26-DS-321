"""
Step 2a: pick a fair sample of activities for the two human raters.

Fair = every repo gets an equal share (so one huge repo can't dominate)
and each share is half commits, half notebook cells.

Run from backend:
    python -m component2.labelling.make_sample
Output:
    component2/data/labelling/rater1.csv  and  rater2.csv  (same items)
"""
import csv
import random
from collections import defaultdict

from component2.config import DATA_DIR
from component2.db import connect

SAMPLE_SIZE = 320      # a few extra in case some items get dropped
MIN_TEXT_LEN = 15      # skip tiny items like "wip" or "x = 1"
SEED = 42              # same seed = same sample every time (reproducible)
OUT_DIR = DATA_DIR / "labelling"


def load_candidates(conn):
    rows = conn.execute("SELECT activity_id, team_id, source, text FROM activities").fetchall()
    pools = defaultdict(list)      # (team_id, source) -> items
    for activity_id, team_id, source, text in rows:
        if len(text.strip()) >= MIN_TEXT_LEN:
            pools[(team_id, source)].append({"activity_id": activity_id, "source": source, "text": text})
    return pools


def pick_sample(pools):
    rng = random.Random(SEED)
    teams = sorted({team for team, _ in pools})
    per_slot = SAMPLE_SIZE // (len(teams) * 2)       # slot = one repo + one source
    chosen, leftovers = [], []
    for team in teams:
        for source in ("commit", "notebook_cell"):
            items = pools.get((team, source), [])
            rng.shuffle(items)
            chosen += items[:per_slot]
            leftovers += items[per_slot:]
    rng.shuffle(leftovers)                            # small repos may not fill their slots
    chosen += leftovers[:SAMPLE_SIZE - len(chosen)]
    rng.shuffle(chosen)                               # mix repos so raters don't see patterns
    return chosen


def write_csv(items, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig opens nicely in Excel
        w = csv.writer(f)
        w.writerow(["item_no", "activity_id", "source", "text", "label", "note"])
        for i, it in enumerate(items, start=1):
            w.writerow([i, it["activity_id"], it["source"], it["text"], "", ""])


def main():
    conn = connect()
    sample = pick_sample(load_candidates(conn))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rater in ("rater1", "rater2"):
        write_csv(sample, OUT_DIR / f"{rater}.csv")
    n_commits = sum(it["source"] == "commit" for it in sample)
    n_teams = len({it["activity_id"].split(":")[1] for it in sample})
    print(f"Sample: {len(sample)} items ({n_commits} commits, {len(sample) - n_commits} cells) "
          f"from {n_teams} repos -> {OUT_DIR}")


if __name__ == "__main__":
    main()