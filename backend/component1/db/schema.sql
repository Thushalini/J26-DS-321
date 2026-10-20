-- PostgreSQL schema for Component 1 (Day 1)
-- Run:  psql -U postgres -c "CREATE DATABASE component1"
--       psql -U postgres -d component1 -f db/schema.sql

CREATE TABLE IF NOT EXISTS projects (
    project_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Chat messages arrive here already structured (who, when, what).
CREATE TABLE IF NOT EXISTS messages (
    message_id       BIGSERIAL PRIMARY KEY,
    project_id       TEXT NOT NULL REFERENCES projects(project_id),
    sender_id        TEXT NOT NULL,
    text             TEXT NOT NULL,
    sent_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    relevance_label  TEXT CHECK (relevance_label IN ('irrelevant', 'relevant_info', 'decision_related')),
    processed        BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_messages_project_time ON messages (project_id, sent_at);

-- Uploaded documents (text is extracted later).
CREATE TABLE IF NOT EXISTS documents (
    document_id  BIGSERIAL PRIMARY KEY,
    project_id   TEXT NOT NULL REFERENCES projects(project_id),
    filename     TEXT NOT NULL,
    uploaded_by  TEXT,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The event log: every blueprint change, never edited, never deleted.
CREATE TABLE IF NOT EXISTS blueprint_events (
    event_id    BIGSERIAL PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(project_id),
    version     INTEGER NOT NULL CHECK (version >= 1),
    event_type  TEXT NOT NULL CHECK (event_type IN (
                    'field_updated', 'decision_recorded',
                    'decision_status_changed', 'scores_updated')),
    payload     JSONB NOT NULL,
    actor       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (project_id, version)
);

-- Make the log append-only: block UPDATE and DELETE.
CREATE OR REPLACE FUNCTION forbid_event_changes() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'blueprint_events is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_events_append_only ON blueprint_events;
CREATE TRIGGER trg_events_append_only
    BEFORE UPDATE OR DELETE ON blueprint_events
    FOR EACH ROW EXECUTE FUNCTION forbid_event_changes();

-- Optional speed-up: a saved copy of the blueprint at a version.
-- The event log stays the source of truth; snapshots can always be rebuilt.
CREATE TABLE IF NOT EXISTS blueprint_snapshots (
    project_id  TEXT NOT NULL REFERENCES projects(project_id),
    version     INTEGER NOT NULL,
    snapshot    JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project_id, version)
);