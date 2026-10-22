"""
Step 3b: Drift detector (FR05, FR06).

For each team and each blueprint decision (target / algorithm / metric):
  1. Collect what the team mentioned in the recent window (mentions.py): commit messages,
     notebook cells, chat AND the code lines each commit added to .py / .R files.
  2. Compare each mentioned term with the planned value using MiniLM
     cosine similarity. A term counts as "on plan" if similarity >= MATCH_AT
     (so "RF" or "RandomForestClassifier" still match "Random Forest").
  3. drift score = share of mentions that are NOT on plan  (0 = on plan, 1 = fully off plan)
  4. severity from the score, plus a one-line "what changed" explanation.

If a team has no blueprint file (public GitHub repos don't), one is INFERRED
from the first 40% of its timeline: "what they did early on = what they planned".

One model or several: a blueprint can also list PARTS, each a folder with its own plan
(e.g. one model per component). Every commit, notebook cell and changed code file then goes to
the part whose folder it is in and is compared with THAT part's plan; chat, and commits spread
over several parts, are compared with the project's plan. Files listed under `ignore` (code that
only lists algorithm names, like keyword lists) are skipped. Without parts it works as before.
A blueprint file that can't be read is NOT replaced: every decision is reported as 'no_plan'
until someone fixes it (api/contract.py describes what a valid blueprint looks like).

Run from backend:
    python -m component2.drift.detect_drift            (all teams, last 30 days of each)
    python -m component2.drift.detect_drift --days 14
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from fnmatch import fnmatch

from component2.api.contract import PART_SEPARATOR, clean_blueprint
from component2.config import BLUEPRINT_DIR
from component2.db import connect, save_drift
from component2.drift.mentions import count_mentions, find_mentions, normalise_target
from component2.embedder import embed
from component2.research_teams import research_team_ids

DIMENSIONS = ("target", "algorithm", "metric")
MATCH_AT = 0.75                 # cosine similarity needed to count as "same thing"
SEVERITY = [(0.20, "none"), (0.40, "minor"), (0.60, "moderate"), (1.01, "significant")]
EARLY_SHARE = 0.40              # first 40% of a timeline is used to infer a blueprint


def parse_ts(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


def severity_of(score):
    for limit, name in SEVERITY:
        if score < limit:
            return name


def texts_of(rows):
    """rows: (ts, text, source, meta). A commit's changed code files are joined into ONE text,
    so a name used in several files of the same commit counts once, like in a commit message."""
    texts, code = [], defaultdict(list)
    for _, text, source, meta in rows:
        if source == "code_change":
            code[json.loads(meta or "{}").get("commit")].append(text)
        else:
            texts.append(text)
    return texts + ["\n".join(files) for files in code.values()]


def load_or_infer_blueprint(team_id, rows):
    """-> (clean blueprint {"plan", "parts", "ignore"}, source)."""
    path = BLUEPRINT_DIR / f"{team_id}.json"
    if path.exists():
        try:
            bp = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"WARNING: {path.name} is not valid JSON - drift reported as 'no_plan' until it is fixed")
            return clean_blueprint(None), "invalid_file"
        return clean_blueprint(bp), "file"

    early = rows[: max(1, int(len(rows) * EARLY_SHARE))]
    seen = count_mentions(texts_of(early))
    planned = {d: (seen[d].most_common(1)[0][0] if seen[d] else None) for d in DIMENSIONS}
    BLUEPRINT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"team_id": team_id, "_inferred_from_early_activity": True,
                                **{d: {"name": planned[d]} for d in DIMENSIONS}}, indent=2),
                    encoding="utf-8")
    return clean_blueprint(planned), "inferred"


def is_ignored(path, ignore):
    return any(path == pattern or path.startswith(pattern + "/") or fnmatch(path, pattern) for pattern in ignore)


def part_of(path, parts):
    """Name of the part whose folder holds this file (the deepest folder wins), or None."""
    inside = [p for p in parts if path == p["path"] or path.startswith(p["path"] + "/")]
    return max(inside, key=lambda p: len(p["path"]))["name"] if inside else None


def split_by_part(rows, parts, ignore):
    """rows: (ts, text, source, meta) -> {part name or None (= the project): rows}, ignored files left out.
    A commit goes to the part most of its (not ignored) files are in; a tie or no files -> the project."""
    groups = defaultdict(list)
    for row in rows:
        _, _, source, meta = row
        meta = json.loads(meta or "{}")
        if source == "commit":
            files = meta.get("files") or []
            kept = [f for f in files if not is_ignored(f, ignore)]
            if files and not kept:
                continue                                  # only touched ignored files
            votes = Counter(part_of(f, parts) for f in kept).most_common(2)
            tie = len(votes) == 2 and votes[0][1] == votes[1][1]
            groups[votes[0][0] if votes and not tie else None].append(row)
            continue
        path = meta.get("file") or meta.get("notebook")
        if path and is_ignored(path, ignore):
            continue
        groups[part_of(path, parts) if path else None].append(row)
    return groups


def planned_names(dim, planned):
    """The plan in the same short form as mentions, e.g. 'Random Forest Classifier' -> {'random forest'}."""
    if dim == "target":
        return {normalise_target(planned)}
    return set(find_mentions(planned)[dim])

def score_dimension(dim, planned, seen: Counter):
    """Returns (drift_score, severity, top_seen_term, explanation)."""
    if not planned:
        return None, "no_plan", None, "Blueprint has no value for this decision."
    if not seen:
        return None, "no_evidence", None, f"Nothing mentioned about it in this window (plan: {planned})."

    terms = list(seen)
    vectors = embed([planned] + terms)
    sims = vectors[1:] @ vectors[0]                    # cosine similarity (vectors are normalised)
    total = sum(seen.values())
    known = planned_names(dim, planned)
    # on plan = same short name, OR close enough in meaning (MiniLM)
    off_plan = {t: seen[t] for t, s in zip(terms, sims) if t not in known and s < MATCH_AT}
    score = round(sum(off_plan.values()) / total, 3)
    sev = severity_of(score)
    top = seen.most_common(1)[0][0]
    if off_plan:
        main_new = max(off_plan, key=off_plan.get)
        why = (f"{score:.0%} of {total} mentions are off plan; most common: '{main_new}' "
               f"({off_plan[main_new]}x) instead of planned '{planned}'.")
    else:
        why = f"All {total} mentions match the planned '{planned}'."
    return score, sev, top, why


def detect_team(conn, team_id, days, planned=None):
    """planned: the app's blueprint (real teams): {dimension: value}, optionally with "parts" and "ignore".
    None = blueprint file or inferred (research teams).
    Returns one result per dimension for the project, plus one per dimension for every part
    (dimension "<part name>::algorithm", and "part" set to the part's name)."""
    rows = conn.execute("SELECT ts, text, source, meta FROM activities WHERE team_id = ? ORDER BY ts",
                        (team_id,)).fetchall()
    if not rows:
        return []
    if planned is None:
        blueprint, bp_source = load_or_infer_blueprint(team_id, rows)
    else:
        blueprint, bp_source = clean_blueprint(planned), "app"
    window_end = parse_ts(rows[-1][0])
    window_start = window_end - timedelta(days=days)
    recent = split_by_part([r for r in rows if parse_ts(r[0]) >= window_start],
                           blueprint["parts"], blueprint["ignore"])

    results = []
    plans = [(None, blueprint["plan"])] + [(p["name"], p["plan"]) for p in blueprint["parts"]]
    for part, plan in plans:
        seen = count_mentions(texts_of(recent.get(part, [])))
        for dim in DIMENSIONS:
            score, sev, top, why = score_dimension(dim, plan[dim], seen[dim])
            results.append({"team_id": team_id, "window_end": rows[-1][0],
                            "dimension": f"{part}{PART_SEPARATOR}{dim}" if part else dim, "part": part,
                            "planned": plan[dim], "seen_top": top, "drift_score": score,
                            "severity": sev, "explanation": why, "blueprint": bp_source})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30, help="size of the recent window")
    ap.add_argument("--team", help="only this team_id")
    args = ap.parse_args()

    conn = connect()
    teams = [args.team] if args.team else research_team_ids(conn)   # real teams: buddy.sync
    for team in teams:
        results = detect_team(conn, team, args.days)
        save_drift(conn, results)
        print(f"\n{team}  (blueprint: {results[0]['blueprint'] if results else '-'})")
        for r in results:
            score = "  -  " if r["drift_score"] is None else f"{r['drift_score']:.2f}"
            print(f"  {r['dimension']:<10} {r['severity']:<12} {score}  {r['explanation']}")


if __name__ == "__main__":
    main()