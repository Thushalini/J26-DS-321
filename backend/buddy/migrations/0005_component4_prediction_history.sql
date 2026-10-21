-- Component 4: one saved prediction per team per day (measured signals only), for the trend line and
-- the "what changed since last time" re-evaluation.
CREATE TABLE IF NOT EXISTS component4_prediction_history (
    team_id       TEXT NOT NULL,
    day           TEXT NOT NULL,
    evaluated_at  TEXT NOT NULL,
    probability   REAL NOT NULL,
    model_version TEXT,
    features      TEXT NOT NULL,
    PRIMARY KEY (team_id, day)
);
