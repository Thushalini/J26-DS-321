-- Each supervisor has a personal code; students enter it to add that supervisor to their team.
ALTER TABLE users ADD COLUMN supervisor_code TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS ix_users_supervisor_code ON users (supervisor_code);
