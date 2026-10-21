"""Adds 5 mock groups ("Mock 1" ... "Mock 5") to one supervisor's list, for demos. Safe to re-run.

Each mock group gets a status dot and a few simulated experiment runs (Component 3 alerts), so the
Summary screen shows different outcomes per group. Everything mock uses team ids starting with "mock__",
so --remove takes all of it out again and touches nothing else.

Run from backend (it writes to whatever database backend/.env points at):
    python -m buddy.seed_mock_groups --supervisor-email supervisor@demo.buddy
    python -m buddy.seed_mock_groups --remove
"""
import argparse

from buddy.codes import new_code
from buddy.store import now, open_conn

PREFIX = "mock__"
GROUPS = [
    {"name": "Mock 1", "status": "attention", "runs": [("HIGH", 0.86, "overfitting"), ("HIGH", 0.81, "overfitting"), ("MEDIUM", 0.52, "stalled")]},
    {"name": "Mock 2", "status": "fine", "runs": [("MEDIUM", 0.48, "premature_evaluation"), ("LOW", 0.21, None), ("LOW", 0.18, None)]},
    {"name": "Mock 3", "status": "good", "runs": [("LOW", 0.12, None), ("LOW", 0.15, None)]},
    {"name": "Mock 4", "status": None, "runs": [("HIGH", 0.9, "data_leakage"), ("LOW", 0.2, None)]},
    {"name": "Mock 5", "status": "attention", "runs": [("HIGH", 0.77, "resource_bottleneck"), ("MEDIUM", 0.55, "resource_bottleneck"), ("HIGH", 0.8, "stalled")]},
]


def _supervisor(conn, email):
    row = conn.execute("SELECT user_id FROM users WHERE email = ? AND role = 'supervisor'", (email.strip().lower(),)).fetchone()
    if not row:
        raise SystemExit(f"No supervisor account with email {email}. Create it first (or run buddy.seed_demo).")
    return row[0]


def _save_runs(conn, team_id, runs):
    from component3.api import store
    store.ensure_schema(conn)
    for index, (level, severity, cause) in enumerate(runs):
        store.save_alert(conn, team_id, {"run_id": f"{team_id}_run{index + 1}", "severity_level": level,
                                         "severity_score": severity, "anomaly_score": severity,
                                         "recommended_cause": cause, "root_cause_probabilities": {}})


def add(conn, supervisor_id):
    for number, group in enumerate(GROUPS, start=1):
        team_id = f"{PREFIX}{number}"
        if not conn.execute("SELECT 1 FROM teams WHERE team_id = ?", (team_id,)).fetchone():
            conn.execute("INSERT INTO teams (team_id, name, is_research, join_code, created_at, sync_status) "
                         "VALUES (?, ?, 0, ?, ?, 'idle')", (team_id, group["name"], new_code(conn), now()))
        conn.execute("INSERT INTO team_members (team_id, user_id, team_role, member_code, git_identities, joined_at) "
                     "VALUES (?, ?, 'supervisor', NULL, '[]', ?) ON CONFLICT (team_id, user_id) DO NOTHING",
                     (team_id, supervisor_id, now()))
        conn.execute("UPDATE team_members SET supervision_status = ?, archived_at = NULL WHERE team_id = ? AND user_id = ?",
                     (group["status"], team_id, supervisor_id))
        conn.commit()
        _save_runs(conn, team_id, group["runs"])
        print(f"  {group['name']:<8} ready ({team_id})")


def remove(conn):
    pattern = f"{PREFIX}%"
    for table in ("component3_alerts", "component4_prediction_history", "component4_competency_sheets",
                  "component4_milestone_results",
                  "panel_reviews", "team_members", "teams"):
        try:
            conn.execute(f"DELETE FROM {table} WHERE team_id LIKE ?", (pattern,))
        except Exception as error:
            conn.raw.rollback()
            print(f"  skipped {table}: {error}")
    conn.commit()
    print("  mock groups removed")


def main():
    parser = argparse.ArgumentParser(description="Add or remove the Mock 1-5 demo groups.")
    parser.add_argument("--supervisor-email")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    if not args.remove and not args.supervisor_email:
        parser.error("--supervisor-email is needed to add the mock groups")
    conn = open_conn()
    try:
        if args.remove:
            remove(conn)
        else:
            add(conn, _supervisor(conn, args.supervisor_email))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
