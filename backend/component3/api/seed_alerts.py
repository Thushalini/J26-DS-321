"""Fill a team with Component 3 alerts from sample runs (for the demo), or find a team's id.

Run from the BACKEND folder (the one that contains main.py):

    cd backend
    python -m component3.api.seed_alerts --find senthuu          # show matching team names and ids
    python -m component3.api.seed_alerts --team <team_id>        # score 8 sample runs, save them for the team

It uses the database set in backend/.env (PostgreSQL if DATABASE_URL is set, otherwise the local SQLite
file) and prints which one. For PostgreSQL it asks you to add --yes, because it creates the table
component3_alerts (if missing) and adds rows there.
"""
import argparse
import sys
from urllib.parse import urlsplit

import pandas as pd

from component2.config import DATABASE_URL, DB_PATH

from ..common import config
from ..predict_live import predict_run, run_from_dataset
from ..scripts.generate_alerts import pick_runs
from . import store


def describe(conn):
    if conn.is_postgres:
        return f"PostgreSQL on {urlsplit(DATABASE_URL.strip()).hostname}"
    return f"local SQLite file {DB_PATH}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--find", metavar="NAME", help="part of a team name")
    g.add_argument("--team", metavar="TEAM_ID", help="the team to fill with sample alerts")
    ap.add_argument("--yes", action="store_true", help="confirm writing to a PostgreSQL database")
    a = ap.parse_args()

    conn = store.connect(create_tables=False)
    print("Database:", describe(conn))

    if a.find:
        rows = conn.execute("SELECT team_id, name FROM teams WHERE lower(name) LIKE ? ORDER BY name",
                            (f"%{a.find.lower()}%",)).fetchall()
        for team_id, name in rows:
            print(f"  {name}   ->   team_id = {team_id}")
        if not rows:
            print("  no team name contains that text")
        return

    if not conn.execute("SELECT 1 FROM teams WHERE team_id = ?", (a.team,)).fetchone():
        sys.exit(f"No team with team_id '{a.team}'. Use --find to look it up.")
    if conn.is_postgres and not a.yes:
        sys.exit("This writes to the PostgreSQL database. Add --yes if that is what you want.")

    store.ensure_schema(conn)
    traj = pd.read_csv(config.TRAJECTORIES_CSV)
    runs = pd.read_csv(config.RUN_LABELS_CSV)
    split = pd.read_csv(config.SPLITS_CSV)
    for run_id in pick_runs(runs, split):
        result = predict_run(run_from_dataset(run_id, traj, runs))
        store.save_alert(conn, a.team, result)
        print(f"  saved {run_id}: {result['severity_level']:<6} {result['title']}")
    print("Done. Open the Alerts page for this team.")


if __name__ == "__main__":
    main()
