"""
Step 6b: System Usability Scale (SUS) score for the Behavior dashboard (target >= 68).

Input: component2/data/evaluation/sus.csv with columns q1 ... q10 (answers 1-5,
1 = strongly disagree, 5 = strongly agree), one row per participant.

Standard SUS scoring (Brooke, 1996):
  odd questions  (positive wording): answer - 1
  even questions (negative wording): 5 - answer
  SUS = sum of the 10 values x 2.5   (0 - 100)

Run from backend:
    python -m component2.evaluation.sus_score
"""
import csv
import statistics

from component2.config import DATA_DIR

SUS_FILE = DATA_DIR / "evaluation" / "sus.csv"


def sus(answers):
    total = 0
    for i, a in enumerate(answers, start=1):
        total += (a - 1) if i % 2 == 1 else (5 - a)
    return total * 2.5


def main():
    with open(SUS_FILE, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    scores = []
    for r in rows:
        answers = [int(r[f"q{i}"]) for i in range(1, 11)]
        if all(1 <= a <= 5 for a in answers):
            scores.append(sus(answers))
    if not scores:
        print("No valid responses.")
        return
    mean = statistics.mean(scores)
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    print(f"Participants: {len(scores)}")
    print("Scores      : " + ", ".join(f"{s:.1f}" for s in scores))
    print(f"Mean SUS    : {mean:.1f}  (SD {sd:.1f})  -> {'meets' if mean >= 68 else 'below'} the 68 target")


if __name__ == "__main__":
    main()