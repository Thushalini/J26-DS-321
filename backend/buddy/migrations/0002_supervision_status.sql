-- A supervisor's own view of each team they supervise: a manual status dot and "removed from my list".
ALTER TABLE team_members ADD COLUMN supervision_status TEXT;
ALTER TABLE team_members ADD COLUMN archived_at TEXT;
