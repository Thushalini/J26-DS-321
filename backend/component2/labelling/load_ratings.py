"""
Step 2c: read the two raters' CSV files, measure how much they agree
(Cohen's Kappa), and build the "gold" labels used to train and test the model.

Workflow:
  1. Put the returned files at  component2/data/labelling/rater1.csv  and  rater2.csv
  2. python -m component2.labelling.load_ratings
  3. If it reports disagreements: open to_resolve.csv, fill the 'final' column
     (talk it through with both raters), save it as resolved.csv, run step 2 again.
"""
import csv

from sklearn.metrics import cohen_kappa_score

from component2.config import LABEL_DIR, STAGES
from component2.db import connect, save_labels

VALID = set(STAGES) | {"other"}


def read_csv_any_encoding(path):
    """Excel sometimes re-saves CSVs in a Windows encoding, so try both."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}")


def read_rater(path):
    labels, problems = {}, []
    for row in read_csv_any_encoding(path):
        label = (row.get("label") or "").strip().lower().replace(" ", "_")
        if label in VALID:
            labels[row["activity_id"]] = label
        else:
            problems.append((row.get("item_no"), label or "(empty)"))
    if problems:
        print(f"  {path.name}: {len(problems)} rows skipped (empty or misspelt), e.g. {problems[:5]}")
    return labels

#round 2
def read_all(rater):
    """Round 1 file plus the round 2 file if it exists (e.g. rater1.csv + rater1_round2.csv)."""
    labels = read_rater(LABEL_DIR / f"{rater}.csv")
    extra = LABEL_DIR / f"{rater}_round2.csv"
    if extra.exists():
        labels.update(read_rater(extra))
    return labels


def round_ids(rater, suffix):
    path = LABEL_DIR / f"{rater}{suffix}.csv"
    return {row["activity_id"] for row in read_csv_any_encoding(path)} if path.exists() else set()


def kappa_meaning(k):
    if k >= 0.80:
        return "almost perfect"
    if k >= 0.60:
        return "substantial"
    if k >= 0.40:
        return "moderate"
    return "weak - the coding guide needs clearer rules"


def main():
    r1 = read_all(LABEL_DIR / "rater1.csv")
    r2 = read_all(LABEL_DIR / "rater2.csv")
    both = sorted(set(r1) & set(r2))
    if not both:
        print("No items labelled by both raters yet.")
        return

    a = [r1[i] for i in both]
    b = [r2[i] for i in both]
    kappa = cohen_kappa_score(a, b)
    agree = sum(x == y for x, y in zip(a, b))
    print(f"Items labelled by both raters: {len(both)}")
    print(f"Raw agreement: {agree}/{len(both)} = {agree / len(both):.1%}")
    print(f"Cohen's Kappa: {kappa:.3f}  ({kappa_meaning(kappa)}; target >= 0.70)")

    for name, ids in (("Round 1", round_ids("rater1", "")), ("Round 2", round_ids("rater1", "_round2"))):
        part = [i for i in both if i in ids]
        if part and len(part) < len(both):
            k = cohen_kappa_score([r1[i] for i in part], [r2[i] for i in part])
            print(f"  {name}: {len(part)} items, Kappa {k:.3f}")


    conn = connect()
    save_labels(conn, "rater1", [(i, r1[i], None) for i in both])
    save_labels(conn, "rater2", [(i, r2[i], None) for i in both])

    # Gold label = where raters agree, plus resolved disagreements
    gold = {i: r1[i] for i in both if r1[i] == r2[i]}
    for resolved_path in sorted(LABEL_DIR.glob("resolved*.csv")):     # resolved.csv, resolved_round2.csv, ...
        for row in read_csv_any_encoding(resolved_path):
            final = (row.get("final") or "").strip().lower().replace(" ", "_")
            if final in VALID:
                gold[row["activity_id"]] = final

    open_items = [i for i in both if i not in gold]
    if open_items:
        texts = dict(conn.execute("SELECT activity_id, text FROM activities").fetchall())
        with open(LABEL_DIR / "to_resolve.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["activity_id", "text", "rater1", "rater2", "final"])
            for i in open_items:
                w.writerow([i, texts.get(i, ""), r1[i], r2[i], ""])
        print(f"{len(open_items)} disagreements to resolve -> {LABEL_DIR / 'to_resolve.csv'}")

    save_labels(conn, "gold", [(i, s, None) for i, s in gold.items()])
    print(f"Gold labels saved: {len(gold)}")


if __name__ == "__main__":
    main()