"""Git collector: read a repository's commit history and turn it into the
commit-related inputs of Component 3.

Uses the plain `git` command (Git must be installed), so it needs no extra Python package.

    python -m backend.component3.collector.git_collector --repo . --since 2026-10-01 --until 2026-10-14

What it measures for a time window (normally the start and end of one experiment run):
  daily_commits               commits per day (days with no commit count as 0)
  commit_gap_days             days between the last commit and the end of the window
  notebook_edit_freq_per_day  commits that touched a notebook (.ipynb) divided by the window length

Honest limits:
  * Merge commits are skipped by default (they are not team work).
  * Git only records COMMITS, not every save of a notebook, so "notebook edits per day"
    is a lower estimate. The training data used higher values than a typical real repo gives.
  * Days are counted in UTC unless you pass utc_offset_hours (Sri Lanka = 5.5).
"""
import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

DAY = 86400.0


def _git(repo, *args):
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout


def to_utc(when) -> datetime:
    """Accept a datetime or an ISO string ('2026-10-01' or '2026-10-01T10:00:00+05:30').
    A value with no time zone is treated as UTC."""
    if isinstance(when, str):
        when = datetime.fromisoformat(when.strip().replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def read_commits(repo, ref="HEAD", include_merges=False):
    """Every commit reachable from `ref` as (unix_timestamp, [changed file paths])."""
    args = ["log", ref, "--pretty=format:@@@%ct", "--name-only"]
    if not include_merges:
        args.insert(2, "--no-merges")
    commits, current = [], None
    for line in _git(repo, *args).splitlines():
        if line.startswith("@@@"):
            current = (int(line[3:]), [])
            commits.append(current)
        elif line.strip() and current is not None:
            current[1].append(line.strip())
    return commits


def collect_git_activity(repo, since, until=None, ref="HEAD", include_merges=False,
                         notebook_suffixes=(".ipynb",), utc_offset_hours=0.0) -> dict:
    """Commit-based inputs for the window [since, until]."""
    repo = Path(repo)
    if _git(repo, "rev-parse", "--is-inside-work-tree").strip() != "true":
        raise ValueError(f"{repo} is not a Git repository.")
    since_dt = to_utc(since)
    until_dt = to_utc(until) if until is not None else datetime.now(timezone.utc)
    if until_dt <= since_dt:
        raise ValueError("'until' must be after 'since'.")
    s, u = since_dt.timestamp(), until_dt.timestamp()
    shift = utc_offset_hours * 3600

    def day_of(ts):
        return (datetime.fromtimestamp(ts + shift, timezone.utc)).date()

    commits = read_commits(repo, ref, include_merges)
    in_window = [(ts, files) for ts, files in commits if s <= ts <= u]

    # commits per day, every day in the window (zeros included)
    first_day, last_day = day_of(s), day_of(u)
    daily = {}
    d = first_day
    while d <= last_day:
        daily[d.isoformat()] = 0
        d += timedelta(days=1)
    for ts, _ in in_window:
        daily[day_of(ts).isoformat()] += 1

    window_days = (u - s) / DAY
    nb = sum(1 for _, files in in_window
             if any(f.lower().endswith(tuple(notebook_suffixes)) for f in files))
    before_end = [ts for ts, _ in commits if ts <= u]
    gap = round((u - max(before_end)) / DAY, 2) if before_end else None

    return {
        "repo": str(repo),
        "since": since_dt.isoformat(), "until": until_dt.isoformat(),
        "window_days": round(window_days, 2),
        "n_commits": len(in_window),
        "n_notebook_commits": nb,
        "daily_commits": daily,
        "commit_gap_days": gap,                       # None = no commit before the window ended
        "notebook_edit_freq_per_day": round(nb / window_days, 3),
        "utc_offset_hours": utc_offset_hours,
    }


def main():
    p = argparse.ArgumentParser(description="Read commit activity from a Git repository.")
    p.add_argument("--repo", default=".", help="path to the repository (default: current folder)")
    p.add_argument("--since", required=True, help="start of the window, e.g. 2026-10-01")
    p.add_argument("--until", help="end of the window (default: now)")
    p.add_argument("--ref", default="HEAD", help="branch or commit to read (default: HEAD)")
    p.add_argument("--utc-offset", type=float, default=0.0,
                   help="hours to add to UTC when assigning commits to days (Sri Lanka: 5.5)")
    p.add_argument("--include-merges", action="store_true")
    a = p.parse_args()
    print(json.dumps(collect_git_activity(a.repo, a.since, a.until, a.ref, a.include_merges,
                                          utc_offset_hours=a.utc_offset), indent=2))


if __name__ == "__main__":
    main()
