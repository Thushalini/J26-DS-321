-- Component 4: the real result a group records after each milestone (self-reported), used to check
-- Buddy's earlier predictions against what actually happened.
CREATE TABLE IF NOT EXISTS component4_milestone_results (
    team_id        TEXT NOT NULL,
    result_id      TEXT NOT NULL,
    milestone      TEXT NOT NULL,
    milestone_date TEXT NOT NULL,
    result         TEXT NOT NULL,
    recorded_by    TEXT,
    recorded_at    TEXT NOT NULL,
    PRIMARY KEY (team_id, result_id)
);
