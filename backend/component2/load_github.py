import json
from datetime import datetime, timezone

from component2.anonymise import MemberResolver


def to_utc(timestamp):
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_noise(commit):
    name = commit["author_name"].lower()
    msg = commit["message"].strip()
    return "[bot]" in name or msg.startswith("Merge ") or msg == ""


def load_commits(repo_dir, resolver):
    commits = json.loads((repo_dir / "commits.json").read_text(encoding="utf-8"))
    rows = []
    for c in commits:
        if is_noise(c):
            continue
        member_id = resolver.get_id(c["author_name"], c["author_email"])
        files = c.get("files_changed", [])
        text = c["message"]
        if files:
            text += "\nFiles: " + ", ".join(files[:15])
        rows.append({
            "activity_id": f"commit:{resolver.team_id}:{c['sha'][:12]}",
            "team_id": resolver.team_id,
            "source": "commit",
            "member_id": member_id,
            "ts": to_utc(c["timestamp"]),
            "text": text,
            "meta": {"files": files[:50], "n_files": len(files),
                     "insertions": c.get("insertions", 0),
                     "deletions": c.get("deletions", 0)},
        })
    for r in rows:                      # scrub after all names are known
        r["text"] = resolver.scrub(r["text"])
    return rows, commits


def last_touch(commits, nb_path, resolver):
    """Who changed this notebook most recently, and when."""
    touching = [c for c in commits if nb_path in c.get("files_changed", []) and not is_noise(c)]
    if not touching:
        return None, None
    latest = max(touching, key=lambda c: to_utc(c["timestamp"]))
    return resolver.get_id(latest["author_name"], latest["author_email"]), to_utc(latest["timestamp"])


def load_notebooks(repo_dir, resolver, commits):
    """Each cell is credited to the commit that wrote it (cell 'sha' from extract_repos).
    Cells without one (older extractions, renamed notebooks) fall back to the notebook's last touch."""
    nb_file = repo_dir / "notebooks.json"
    if not nb_file.exists():
        return []
    notebooks = json.loads(nb_file.read_text(encoding="utf-8"))
    by_sha = {c["sha"]: c for c in commits if not is_noise(c)}
    rows = []
    for nb in notebooks:
        last_member, last_ts = last_touch(commits, nb["path"], resolver)
        for i, cell in enumerate(nb["cells"]):
            source = cell["source"].strip()
            if not source:
                continue
            writer = by_sha.get(cell.get("sha"))
            if writer:
                member_id, ts = resolver.get_id(writer["author_name"], writer["author_email"]), to_utc(writer["timestamp"])
                dated_by = "cell_history"
            else:
                member_id, ts = last_member, last_ts
                dated_by = "last_touch"
            if ts is None:
                continue
            rows.append({
                "activity_id": f"cell:{resolver.team_id}:{nb['path']}:{i}",
                "team_id": resolver.team_id,
                "source": "notebook_cell",
                "member_id": member_id,
                "ts": ts,
                "text": resolver.scrub(source[:2000]),
                "meta": {"notebook": nb["path"], "cell_type": cell["cell_type"], "cell_index": i,
                         "dated_by": dated_by},
            })
    return rows


def load_code_changes(resolver, commits):
    """One activity per code file a commit changed, holding the lines it ADDED (.py / .R),
    so the classifier and the drift detector see the code itself, not only the commit message.
    Commits extracted before code_added existed simply give none (re-extract with --refresh)."""
    rows = []
    for c in commits:
        if is_noise(c):
            continue
        member_id = resolver.get_id(c["author_name"], c["author_email"])
        for path, added in (c.get("code_added") or {}).items():
            rows.append({
                "activity_id": f"code:{resolver.team_id}:{c['sha'][:12]}:{path}",
                "team_id": resolver.team_id,
                "source": "code_change",
                "member_id": member_id,
                "ts": to_utc(c["timestamp"]),
                "text": resolver.scrub(f"File: {path}\n{added}"),
                "meta": {"file": path, "commit": c["sha"][:12], "cell_type": "code"},
            })
    return rows


def load_repo(repo_dir):
    resolver = MemberResolver(team_id=repo_dir.name)
    commit_rows, commits = load_commits(repo_dir, resolver)
    cell_rows = load_notebooks(repo_dir, resolver, commits)
    code_rows = load_code_changes(resolver, commits)
    return commit_rows, cell_rows, code_rows