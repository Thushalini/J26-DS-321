"""
Step 6a: Compare the role tracker with what team members SAY they worked on (objective O4).

In our team everyone works across the whole lifecycle (each member trains their own
model), so a member's "role" is the stage they are mainly in at a given time.
We therefore ask each member which stage they were MAINLY in for each recent week,
and check whether the tracker found the same stage for that member in that week.

Input: component2/data/evaluation/self_report.csv  (private - contains real names)
  columns: name, git_emails, then one column per ISO week, e.g. 2026-W39, 2026-W40, ...
  - git_emails: every email the person commits with, separated by ;
  - week cells: the stage name as in the form (e.g. "Model development"), or empty if no work
Results are saved WITHOUT names.

Run from backend:
    python -m component2.evaluation.self_report_check --team own__J26-DS-321
"""
import argparse
import csv
import re

from component2.anonymise import MemberResolver
from component2.config import DATA_DIR, RAW_DATA_DIR, STAGES
from component2.db import connect
from component2.load_github import load_commits
from component2.roles.role_tracker import iso_week, stage_labels, weekly_stage_mix

EVAL_DIR = DATA_DIR / "evaluation"
WEEK = re.compile(r"\d{4}-W\d{2}")
LABEL_TO_STAGE = {
    "data cleaning": "data_cleaning", "eda": "eda", "exploratory data analysis": "eda",
    "feature engineering": "feature_engineering", "model development": "model_development",
    "model evaluation": "model_evaluation", "deployment / mlops": "deployment_mlops",
    "deployment": "deployment_mlops", "documentation": "documentation",
}


def to_stage(text):
    t = (text or "").strip().lower()
    if not t:
        return None
    return LABEL_TO_STAGE.get(t, t.replace(" ", "_") if t.replace(" ", "_") in STAGES else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    args = ap.parse_args()

    # Rebuild the same name/email -> code mapping that was used when the repo was loaded
    resolver = MemberResolver(team_id=args.team)
    load_commits(RAW_DATA_DIR / args.team, resolver)

    conn = connect()
    tracked = {(m, w): d for m, w, d in conn.execute(
        "SELECT member_id, week, dominant_stage FROM role_trends WHERE team_id = ?", (args.team,)).fetchall()}
    members_in_db = {m for m, _ in tracked}

    # Also the RAW weekly role (no EMA smoothing), to show what smoothing changes
    work = [(aid, m, iso_week(ts)) for aid, m, ts in conn.execute(
        "SELECT activity_id, member_id, ts FROM activities WHERE team_id = ? AND source != 'chat'",
        (args.team,)).fetchall()]
    raw_mix = weekly_stage_mix(work, stage_labels(conn), sorted(members_in_db))
    raw = {(m, w): max(mix, key=mix.get) for w, per in raw_mix.items() for m, mix in per.items()
           if sum(mix.values()) > 0}

    with open(EVAL_DIR / "self_report.csv", newline="", encoding="utf-8-sig") as f:
        reports = list(csv.DictReader(f))
    weeks = [c for c in (reports[0].keys() if reports else []) if WEEK.fullmatch(c.strip())]

    rows = []
    for r in reports:
        emails = [e.strip().lower() for e in (r.get("git_emails") or "").split(";") if e.strip()]
        codes = [resolver.by_email.get(e) for e in emails] + [resolver.by_name.get(r["name"].strip().lower())]
        code = next((c for c in codes if c in members_in_db), None)
        if code is None:
            print("  skipped one response: no commits found for that name/emails (check the aliases file)")
            continue
        for week in weeks:
            said = to_stage(r[week])
            found = tracked.get((code, week.strip()))
            if said is None or found is None:
                continue                      # no self-report or no tracked work that week
            raw_found = raw.get((code, week.strip()))
            rows.append({"member_id": code, "week": week.strip(), "self_report": said,
                         "tracker": found, "match": int(said == found),
                         "tracker_raw": raw_found, "match_raw": int(said == raw_found)})

    if not rows:
        print("No weeks to compare (check week column names match role_trends weeks, e.g. 2026-W40).")
        return
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "self_report_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'member':<10} {'week':<9} {'member says':<20} {'tracker (EMA)':<20} {'tracker (raw week)':<20}")
    for r in rows:
        print(f"{r['member_id']:<10} {r['week']:<9} {r['self_report']:<20} "
              f"{r['tracker'] + (' *' if r['match'] else ''):<20} {str(r['tracker_raw']) + (' *' if r['match_raw'] else ''):<20}")
    n = len(rows)
    ok, ok_raw = sum(r["match"] for r in rows), sum(r["match_raw"] for r in rows)
    print("(* = same as what the member said)")
    print(f"\nMember-weeks compared: {n}   from {len({r['member_id'] for r in rows})} members")
    print(f"Weekly agreement, smoothed (EMA) roles: {ok}/{n} = {ok / n:.0%}")
    print(f"Weekly agreement, raw weekly roles    : {ok_raw}/{n} = {ok_raw / n:.0%}")
    print(f"Saved (no names) -> {out}")


if __name__ == "__main__":
    main()