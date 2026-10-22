"""
Round 2 labelling sample: get MORE gold examples for the stages the model is weak on.

Round 1 was a fair random sample (stratified by repo), so rare stages like
model_evaluation got very few items. Here we pick NEW, not-yet-labelled activities
that the current model THINKS belong to the rare stages, plus some uncertain and
some random ones (targeted sampling, a standard active-learning step).
The raters still decide the real label - the model's guess is never shown to them.

Run from backend (after train_model):
    python -m component2.labelling.make_sample_round2
Output:
    component2/data/labelling/rater1_round2.csv and rater2_round2.csv (same items)
"""
import csv
import random
from collections import Counter, defaultdict

from component2.config import LABEL_DIR
from component2.db import connect

SEED = 7
MIN_TEXT_LEN = 15
MAX_SHARE_PER_REPO = 0.15        # no repo may supply more than 15% of the sample
QUOTAS = {                       # predicted stage -> how many items
    "model_evaluation": 70,
    "deployment_mlops": 60,
    "model_development": 50,
    "feature_engineering": 40,
}
N_UNCERTAIN = 40                 # lowest-confidence items, any stage
N_RANDOM = 40                    # plain random items, keeps the sample honest


def main():
    rng = random.Random(SEED)
    conn = connect()
    labelled = {aid for (aid,) in conn.execute(
        "SELECT DISTINCT activity_id FROM stage_labels WHERE labeller IN ('rater1', 'rater2', 'gold')").fetchall()}
    with open(LABEL_DIR / "rater1.csv", newline="", encoding="utf-8-sig") as f:      # everything shown in round 1
        labelled |= {row["activity_id"] for row in csv.DictReader(f)}
    rows = conn.execute(
        "SELECT a.activity_id, a.team_id, a.source, a.text, l.stage, l.confidence FROM activities a "
        "JOIN stage_labels l ON l.activity_id = a.activity_id AND l.labeller = 'model' "
        "WHERE a.source != 'chat'").fetchall()            # chat is simulated, so never used as gold
    pool = [r for r in rows if r[0] not in labelled and len(r[3].strip()) >= MIN_TEXT_LEN]
    rng.shuffle(pool)

    total = sum(QUOTAS.values()) + N_UNCERTAIN + N_RANDOM
    cap = max(5, int(total * MAX_SHARE_PER_REPO))
    per_repo, chosen, taken = Counter(), [], set()

    def take(candidates, n):
        got = 0
        for r in candidates:
            if got >= n:
                break
            if r[0] in taken or per_repo[r[1]] >= cap:
                continue
            chosen.append(r)
            taken.add(r[0])
            per_repo[r[1]] += 1
            got += 1
        return got

    report = {}
    for stage, n in QUOTAS.items():
        report[stage] = take([r for r in pool if r[4] == stage], n)
    report["uncertain"] = take(sorted(pool, key=lambda r: r[5]), N_UNCERTAIN)
    report["random"] = take(pool, N_RANDOM)

    rng.shuffle(chosen)                                   # raters must not see any grouping
    for rater in ("rater1", "rater2"):
        with open(LABEL_DIR / f"{rater}_round2.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["item_no", "activity_id", "source", "text", "label", "note"])
            for i, r in enumerate(chosen, start=1):
                w.writerow([i, r[0], r[2], r[3], "", ""])

    print(f"Round 2 sample: {len(chosen)} new items from {len(per_repo)} repos (max {cap} per repo)")
    for k, v in report.items():
        print(f"  {k:<20} {v}")
    print(f"Saved -> {LABEL_DIR / 'rater1_round2.csv'} and rater2_round2.csv")


if __name__ == "__main__":
    main()