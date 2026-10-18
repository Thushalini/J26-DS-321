"""Prepare the database for a demo (safe to run more than once; never changes research results).

  1. Creates the app tables (users, teams, team_members, chat_messages, blueprints, feedback).
  2. Registers each Phase 1 research team in `teams` with is_research = 1, KEEPING its team_id,
     so none of the existing activities / labels / drift / role rows change.
  3. Creates demo staff accounts (if they don't exist yet):
        supervisor@demo.buddy   (role supervisor)
        examiner@demo.buddy     (role examiner - sees the research dataset + all feedback)

Run from backend:
    python -m buddy.seed_demo
    python -m buddy.seed_demo --password "something-better"
"""
import argparse

from buddy.auth import hash_password
from buddy.store import new_id, now, open_conn

DEMO_ACCOUNTS = [("supervisor@demo.buddy", "Demo Supervisor", "supervisor"),
                 ("examiner@demo.buddy", "Demo Examiner", "examiner")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--password", default="buddy-demo-2026", help="password for the demo staff accounts")
    args = ap.parse_args()

    conn = open_conn()
    known = {t for (t,) in conn.execute("SELECT team_id FROM teams").fetchall()}
    research = [t for (t,) in conn.execute("SELECT DISTINCT team_id FROM activities ORDER BY team_id").fetchall()
                if t not in known]
    for team_id in research:
        repo = team_id.split("__", 1)[-1]
        conn.execute("INSERT INTO teams (team_id, name, is_research, created_at, sync_status) "
                     "VALUES (?, ?, 1, ?, 'done')", (team_id, repo, now()))
    conn.commit()
    print(f"Research teams registered now: {len(research)}  (already registered: {len(known)})")

    for email, name, role in DEMO_ACCOUNTS:
        if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            print(f"  {email:<24} already exists (password unchanged)")
            continue
        conn.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
                     (new_id("u"), email, name, role, hash_password(args.password), now()))
        print(f"  {email:<24} created, password: {args.password}")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
