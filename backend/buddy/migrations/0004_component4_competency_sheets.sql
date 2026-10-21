-- Component 4: each group member's parsed score sheet (module code, credits, grade), saved so the
-- "who could help" suggestion can be re-run without uploading again. Never shown to anyone.
CREATE TABLE IF NOT EXISTS component4_competency_sheets (
    team_id     TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    modules     TEXT NOT NULL,
    uploaded_by TEXT,
    uploaded_at TEXT NOT NULL,
    PRIMARY KEY (team_id, user_id)
);
