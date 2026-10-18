"""Which teams belong to the RESEARCH dataset (Phase 1: the 16 public repos).

Real teams that sign up in the app (Phase 2) are stored in the `teams` table with
is_research = 0. The research scripts (simulate_chat, detect_drift, role_tracker,
make_sample, ...) must only ever touch research teams, so their results stay
reproducible and no simulated chat is added to a real team.

Any team that has activities but is NOT in the teams table counts as research
(that is how all the original data was loaded, before the teams table existed).
"""


def _teams_table_exists(conn):
    if conn.is_postgres:
        return conn.execute("SELECT to_regclass('public.teams') IS NOT NULL").fetchone()[0]
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'teams'").fetchone() is not None


def live_team_ids(conn):
    if not _teams_table_exists(conn):
        return set()
    return {t for (t,) in conn.execute("SELECT team_id FROM teams WHERE is_research = 0").fetchall()}


def research_team_ids(conn):
    live = live_team_ids(conn)
    return [t for (t,) in conn.execute("SELECT DISTINCT team_id FROM activities ORDER BY team_id").fetchall()
            if t not in live]
