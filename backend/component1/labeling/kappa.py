#!/usr/bin/env python3
"""Cohen's Kappa between two raters. Standard library only.

    python -m component1.labeling.kappa component1/data/labeling_out/sheet_rater_a.csv component1/data/labeling_out/sheet_rater_b.csv

Both files must have a msg_id column and a label column (default: intent).
Rows where either rater left the label blank are skipped.
Disagreements are saved to a CSV so the two raters can discuss them afterwards.
"""
import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path


def cohen_kappa(a, b):
    """Kappa for two equal-length lists of labels. Returns nan if it is undefined."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty lists of the same length")
    n = len(a)
    labels = set(a) | set(b)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if expected == 1.0:  # both raters used one single label for everything
        return float("nan")
    return (observed - expected) / (1 - expected)


def interpret(k):
    """Landis and Koch (1977) bands, the same ones your report cites."""
    if math.isnan(k):
        return "undefined (both raters used a single label)"
    if k < 0:
        return "poor (worse than chance)"
    if k <= 0.20:
        return "slight"
    if k <= 0.40:
        return "fair"
    if k <= 0.60:
        return "moderate"
    if k <= 0.80:
        return "substantial"
    return "almost perfect"


def read_labels(path, id_col, label_col):
    labels = {}
    # utf-8-sig: Excel's "CSV UTF-8" adds a BOM that would otherwise stick to the first header
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            value = (row.get(label_col) or "").strip().lower()
            if value:
                labels[row[id_col]] = value
    return labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rater_a")
    ap.add_argument("rater_b")
    ap.add_argument("--id-col", default="msg_id")
    ap.add_argument("--label-col", default="intent")
    ap.add_argument("--out", default="component1/data/disagreements.csv",
                    help="where to save disagreements (default: component1/data/disagreements.csv)")
    args = ap.parse_args(argv)

    a = read_labels(args.rater_a, args.id_col, args.label_col)
    b = read_labels(args.rater_b, args.id_col, args.label_col)
    shared = sorted(set(a) & set(b))
    if not shared:
        print("No rows labeled by both raters.")
        return 1

    la = [a[i] for i in shared]
    lb = [b[i] for i in shared]
    k = cohen_kappa(la, lb)
    agree = sum(x == y for x, y in zip(la, lb)) / len(shared)

    print(f"Items labeled by both raters: {len(shared)}")
    print(f"Raw agreement: {agree:.1%}")
    print(f"Cohen's Kappa: {k:.3f}  ->  {interpret(k)}")
    print("Target in the report: 0.70 or higher" + ("  [MET]" if k >= 0.70 else "  [NOT MET]"))

    labels = sorted(set(la) | set(lb))
    counts = Counter(zip(la, lb))
    print("\nRows = rater A, columns = rater B")
    print(" " * 16 + "".join(f"{l[:12]:>14}" for l in labels))
    for x in labels:
        print(f"{x[:14]:<16}" + "".join(f"{counts[(x, y)]:>14}" for y in labels))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([args.id_col, "rater_a", "rater_b"])
        for i, x, y in zip(shared, la, lb):
            if x != y:
                w.writerow([i, x, y])
    print(f"\nDisagreements saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())