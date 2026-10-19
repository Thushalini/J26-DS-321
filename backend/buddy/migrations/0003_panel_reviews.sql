-- Panel review: one temporary, read-only look at another team per supervisor, replaced by the next one.
CREATE TABLE IF NOT EXISTS panel_reviews (
    user_id    TEXT PRIMARY KEY,
    team_id    TEXT NOT NULL,
    started_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
