"""Live sync for a REAL team (Phase 2): the same Component 2 method, different data source.

  research teams (Phase 1)                 real teams (this file)
  ---------------------------------        ------------------------------------------
  search_repos + extract_repos scripts  -> clone the team's own repo link
  MemberResolver (name/email hash)      -> LinkedResolver: commits -> signed-in members
  simulated chat CSV + load_chat        -> in-app chat (chat_messages, written on send)
  inferred blueprint JSON               -> blueprint the team enters (until C1 provides it)

Steps: clone -> extract commits + notebook cells -> load as activities -> classify
(keyword baseline + trained MiniLM model) -> drift vs blueprint -> weekly roles.
"""
import json
import os
import re
import tempfile
import traceback
from pathlib import Path

import numpy as np

from component2.anonymise import MemberResolver, make_code
from component2.api.contract import clean_plan
from component2.api.service import _labels_for_team
from component2.classify.keyword_baseline import classify
from component2.collect.extract_repos import clone_repo, extract_commits, extract_notebooks, force_delete, save_json
from component2.config import MODEL_DIR
from component2.db import connect, save_activities, save_drift, save_labels, save_roles
from component2.load_github import is_noise, load_code_changes, load_commits, load_notebooks

from buddy.store import now, one_as_dict, rows_as_dicts

GITHUB_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+?(\.git)?/?$")
STALE_AFTER_MINUTES = 30            # a "running" sync older than this is treated as crashed


def member_code(team_id, user_id):
    """Stable anonymous code for a signed-in member (same format as research codes)."""
    return make_code(f"{team_id}|user:{user_id}")


class LinkedResolver(MemberResolver):
    """Like MemberResolver, but a commit whose author name/email a member has linked
    gets that member's code. Unlinked authors still get an anonymous hash code."""

    def __init__(self, team_id, identities):
        super().__init__(team_id)
        self.identities = identities                     # lower-case name or email -> member code

    def get_id(self, name, email=""):
        n, e = (name or "").strip().lower(), (email or "").strip().lower()
        code = self.identities.get(e) or self.identities.get(n)
        if not code:
            return super().get_id(name, email)
        if n:
            self.by_name[n] = code
            self.real_names.add(n)
        if e:
            self.by_email[e] = code
        return code


def identity_map(conn, team_id):
    rows = conn.execute("SELECT m.member_code, m.git_identities, u.name, u.email FROM team_members m "
                        "JOIN users u ON u.user_id = m.user_id "
                        "WHERE m.team_id = ? AND m.member_code IS NOT NULL", (team_id,)).fetchall()
    ids = {}
    for code, extra, name, email in rows:
        for key in [name, email] + json.loads(extra or "[]"):
            if key and key.strip():
                ids[key.strip().lower()] = code
    return ids


def clone_url(repo_url):
    """Private repos work if GITHUB_TOKEN (in backend/.env) can read them."""
    token = os.getenv("GITHUB_TOKEN", "")
    url = repo_url.rstrip("/")
    if not url.endswith(".git"):
        url += ".git"
    if token:
        url = url.replace("https://github.com/", f"https://x-access-token:{token}@github.com/")
    return url


def _safe(msg):
    token = os.getenv("GITHUB_TOKEN", "")
    return msg.replace(token, "***") if token else msg


def set_status(conn, team_id, status, message, finished=False):
    if finished:
        conn.execute("UPDATE teams SET sync_status = ?, sync_message = ?, last_synced = ? WHERE team_id = ?",
                     (status, message, now(), team_id))
    else:
        conn.execute("UPDATE teams SET sync_status = ?, sync_message = ? WHERE team_id = ?",
                     (status, message, team_id))
    conn.commit()


def load_repo_activities(conn, team_id, repo_url):
    """Clone, extract and save the repo's commits, notebook cells and code changes.
    Returns (commits, cells, code files)."""
    with tempfile.TemporaryDirectory(prefix="buddy_sync_") as tmp:
        clone_path, raw_dir = Path(tmp) / "repo", Path(tmp) / "raw"
        raw_dir.mkdir()
        try:
            clone_repo(clone_url(repo_url), clone_path)
        except Exception:
            raise RuntimeError("Could not clone the repository. Check the link, or that GITHUB_TOKEN can read it.")
        save_json(extract_commits(clone_path), raw_dir / "commits.json")
        save_json(extract_notebooks(clone_path), raw_dir / "notebooks.json")
        force_delete(clone_path)                 # Windows: .git files are read-only

        resolver = LinkedResolver(team_id, identity_map(conn, team_id))
        commit_rows, commits = load_commits(raw_dir, resolver)
        cell_rows = load_notebooks(raw_dir, resolver, commits)
        code_rows = load_code_changes(resolver, commits)
        linked = set(resolver.identities.values())
        unlinked = {}
        # Use the codes the saved commits actually got: resolving again here (merge commits included)
        # could re-map a name to a linked member and hide authors that are really unlinked
        for c, row in zip([c for c in commits if not is_noise(c)], commit_rows):
            code = row["member_id"]
            if code not in linked:
                a = unlinked.setdefault(code, {"member_code": code, "name": c["author_name"],
                                               "email": c["author_email"], "commits": 0})
                a["commits"] += 1

    # Replace the team's previous repo snapshot (cells can move / disappear between syncs); chat is kept
    old = [aid for (aid,) in conn.execute("SELECT activity_id FROM activities WHERE team_id = ? "
                                          "AND source != 'chat'", (team_id,)).fetchall()]
    for aid in old:
        conn.execute("DELETE FROM stage_labels WHERE activity_id = ?", (aid,))
    conn.execute("DELETE FROM activities WHERE team_id = ? AND source != 'chat'", (team_id,))
    conn.commit()
    save_activities(conn, commit_rows + cell_rows + code_rows)
    conn.execute("UPDATE teams SET unlinked_authors = ? WHERE team_id = ?",
                 (json.dumps(sorted(unlinked.values(), key=lambda a: -a["commits"])), team_id))
    conn.commit()
    return len(commit_rows), len(cell_rows), len(code_rows)


def label_team(conn, team_id):
    """Keyword baseline for every activity, then the trained Step 2d model if it exists."""
    rows = conn.execute("SELECT activity_id, source, text, meta FROM activities WHERE team_id = ? "
                        "ORDER BY activity_id", (team_id,)).fetchall()
    if not rows:
        return False
    save_labels(conn, "baseline", [(aid, *classify(text, json.loads(meta or "{}").get("cell_type")))
                                   for aid, _, text, meta in rows])
    model_file = MODEL_DIR / "stage_classifier.joblib"
    if not model_file.exists():
        return False
    import joblib
    from component2.classify.train_model import model_input, structure_features
    from component2.embedder import embed

    ids = [r[0] for r in rows]
    vectors = embed([model_input(source, text) for _, source, text, _ in rows])
    X = np.hstack([vectors, structure_features(conn, ids)])
    clf = joblib.load(model_file)
    probs = clf.predict_proba(X)
    stages = clf.classes_[probs.argmax(axis=1)]
    save_labels(conn, "model", [(ids[n], stages[n], round(float(probs[n].max()), 3)) for n in range(len(ids))])
    return True


def blueprint_plan(conn, team_id):
    bp = one_as_dict(conn.execute("SELECT target, algorithm, metric FROM blueprints WHERE team_id = ?", (team_id,)))
    return clean_plan(bp)


def analyse_team(conn, team_id, days=30):
    """Drift + roles from whatever activities the team has (repo + chat)."""
    from component2.drift.detect_drift import detect_team
    from component2.roles.role_tracker import track_team

    conn.execute("DELETE FROM drift_results WHERE team_id = ?", (team_id,))
    save_drift(conn, detect_team(conn, team_id, days, planned=blueprint_plan(conn, team_id)))
    conn.execute("DELETE FROM role_trends WHERE team_id = ?", (team_id,))
    save_roles(conn, track_team(conn, team_id, _labels_for_team(conn, team_id)))


def run_sync(team_id):
    """Runs in the background after POST /app/teams/{id}/sync. Progress is written to the teams row."""
    conn = connect(create_tables=False)
    try:
        team = one_as_dict(conn.execute("SELECT * FROM teams WHERE team_id = ?", (team_id,)))
        summary = []
        if team.get("repo_url"):
            set_status(conn, team_id, "running", "Cloning the repository and reading commits + notebooks...")
            n_commits, n_cells, n_code = load_repo_activities(conn, team_id, team["repo_url"])
            summary.append(f"{n_commits} commits, {n_cells} notebook cells, {n_code} changed code files")
        n_chat = conn.execute("SELECT COUNT(*) FROM activities WHERE team_id = ? AND source = 'chat'",
                              (team_id,)).fetchone()[0]
        summary.append(f"{n_chat} chat messages")

        set_status(conn, team_id, "running", "Classifying every activity into lifecycle stages...")
        if not label_team(conn, team_id):
            summary.append("keyword labels only (trained model not found)")

        set_status(conn, team_id, "running", "Checking plan drift and member roles...")
        analyse_team(conn, team_id)
        set_status(conn, team_id, "done", "Analysed " + ", ".join(summary) + ".", finished=True)
    except Exception as e:
        traceback.print_exc()
        try:
            conn.raw.rollback()
        except Exception:
            pass
        set_status(conn, team_id, "error", _safe(str(e)) or "Sync failed.", finished=True)
    finally:
        conn.close()

