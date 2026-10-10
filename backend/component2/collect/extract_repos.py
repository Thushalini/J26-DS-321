"""
Component 2 - Data collection, part 2: clone each candidate repo and save
its commit history and notebook cells as JSON.

Run from the backend folder (after search_repos):
    python -m component2.collect.extract_repos
Output (one folder per repo):
    component2/data/raw_data/<owner>__<repo>/commits.json
    component2/data/raw_data/<owner>__<repo>/notebooks.json
"""
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import git  # GitPython
import nbformat

from component2.config import CANDIDATES_FILE, CLONE_DIR, RAW_DATA_DIR

MAX_REPOS = 25


def safe_name(full_name):
    return full_name.replace("/", "__")


def force_delete(path):
    """Windows marks files inside .git as read-only, so a normal delete fails.
    This makes them writable first, then deletes."""
    def make_writable_and_retry(func, p, exc_info):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if path.exists():
        shutil.rmtree(path, onerror=make_writable_and_retry)


def clone_repo(clone_url, dest_path):
    force_delete(dest_path)
    subprocess.run(["git", "clone", "--quiet", clone_url, str(dest_path)],
                   check=True, timeout=300)


def extract_commits(repo_path):
    repo = git.Repo(repo_path)
    commits = []
    for commit in repo.iter_commits():
        try:
            stats = commit.stats.total
            files_changed = list(commit.stats.files.keys())
        except Exception:
            stats, files_changed = {}, []
        commits.append({
            "sha": commit.hexsha,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "timestamp": commit.committed_datetime.isoformat(),
            "message": commit.message.strip(),
            "files_changed": files_changed,
            "insertions": stats.get("insertions", 0),
            "deletions": stats.get("deletions", 0),
        })
    repo.close()  # release file handles so Windows can delete the clone
    return commits


def extract_notebooks(repo_path):
    notebooks = []
    for nb_path in Path(repo_path).rglob("*.ipynb"):
        if ".ipynb_checkpoints" in str(nb_path):
            continue
        try:
            nb = nbformat.read(nb_path, as_version=4)
        except Exception as e:
            print(f"    Skipped unreadable notebook {nb_path.name}: {e}")
            continue
        cells = [{"cell_type": c.cell_type, "source": c.source} for c in nb.cells]
        notebooks.append({
            "path": nb_path.relative_to(repo_path).as_posix(),   # forward slashes, like git
            "n_cells": len(cells),
            "cells": cells,
        })
    return notebooks


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    with open(CANDIDATES_FILE, encoding="utf-8") as f:
        candidates = json.load(f)

    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    total = min(len(candidates), MAX_REPOS)

    for i, repo_info in enumerate(candidates[:MAX_REPOS], start=1):
        full_name = repo_info["full_name"]
        name = safe_name(full_name)
        out_path = RAW_DATA_DIR / name
        if (out_path / "commits.json").exists():
            print(f"[{i}/{total}] {full_name}: already extracted, skipping")
            continue
        print(f"[{i}/{total}] Processing {full_name}")

        clone_path = CLONE_DIR / name
        try:
            clone_repo(repo_info["clone_url"], clone_path)
        except Exception as e:
            print(f"  Clone failed: {e}")
            continue

        out_path.mkdir(parents=True, exist_ok=True)
        try:
            commits = extract_commits(clone_path)
            save_json(commits, out_path / "commits.json")
            print(f"  Extracted {len(commits)} commits")
        except Exception as e:
            print(f"  Commit extraction failed: {e}")

        try:
            notebooks = extract_notebooks(clone_path)
            save_json(notebooks, out_path / "notebooks.json")
            print(f"  Extracted {len(notebooks)} notebooks")
        except Exception as e:
            print(f"  Notebook extraction failed: {e}")

        force_delete(clone_path)

    print(f"\nDone. Raw data saved under {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()