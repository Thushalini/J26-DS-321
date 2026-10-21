"""
Step 6a: Extract a LOCAL git repository (e.g. our own J26-DS-321 repo) into
raw_data, in exactly the same format as extract_repos.py, so run_step1 can load it.

Run from backend (".." = the repo root one level up):
    python -m component2.collect.extract_local .. --name own__J26-DS-321
"""
import argparse
import json
import subprocess
from pathlib import Path

from component2.collect.extract_repos import extract_code_added
from component2.config import RAW_DATA_DIR

RS, US = "\x1e", "\x1f"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


def extract_commits(repo):
    out = git(repo, "log", "--no-renames", "--numstat", f"--format={RS}%H{US}%an{US}%ae{US}%aI{US}%B{US}")
    code_added = extract_code_added(repo)
    commits = []
    for record in out.split(RS)[1:]:
        parts = record.split(US)
        if len(parts) < 6:
            continue
        sha, name, email, date, message, numstat = parts[:6]
        files, ins, dels = [], 0, 0
        for line in numstat.strip().splitlines():
            bits = line.split("\t")
            if len(bits) == 3:
                files.append(bits[2])
                ins += int(bits[0]) if bits[0].isdigit() else 0     # '-' for binary files
                dels += int(bits[1]) if bits[1].isdigit() else 0
        commits.append({"sha": sha, "author_name": name, "author_email": email, "timestamp": date,
                        "message": message.strip(), "files_changed": files,
                        "insertions": ins, "deletions": dels, "code_added": code_added.get(sha, {})})
    return commits


def extract_notebooks(repo):
    """Only notebooks tracked by git (skips .venv, node_modules, ignored data folders)."""
    notebooks = []
    for rel in git(repo, "ls-files", "*.ipynb").splitlines():
        try:
            nb = json.loads((Path(repo) / rel).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cells = []
        for c in nb.get("cells", []):
            src = c.get("source", "")
            cells.append({"cell_type": c.get("cell_type", "code"),
                          "source": "".join(src) if isinstance(src, list) else src})
        notebooks.append({"path": rel, "n_cells": len(cells), "cells": cells})
    return notebooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="path to the local git repository")
    ap.add_argument("--name", required=True, help="team_id to use, e.g. own__J26-DS-321")
    args = ap.parse_args()

    out = RAW_DATA_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    commits = extract_commits(args.repo)
    notebooks = extract_notebooks(args.repo)
    (out / "commits.json").write_text(json.dumps(commits, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "notebooks.json").write_text(json.dumps(notebooks, indent=2, ensure_ascii=False), encoding="utf-8")
    authors = {c["author_email"] for c in commits}
    print(f"{args.name}: {len(commits)} commits by {len(authors)} author emails, {len(notebooks)} notebooks -> {out}")


if __name__ == "__main__":
    main()