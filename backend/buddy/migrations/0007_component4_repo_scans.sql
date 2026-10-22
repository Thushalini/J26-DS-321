-- Component 4: the last reading of each group's key repo files (counts and flags only, never file contents),
-- so readiness and reproducibility can judge content without fetching from GitHub on every request.
CREATE TABLE IF NOT EXISTS component4_repo_scans (
    team_id    TEXT PRIMARY KEY,
    repo_url   TEXT NOT NULL,
    scanned_at TEXT NOT NULL,
    files      TEXT NOT NULL,
    evidence   TEXT NOT NULL
);
