-- Component 2: multi-model projects. `parts` = JSON list of {name, path, target, algorithm, metric}
-- (one plan per folder, e.g. per component); `ignore_paths` = JSON list of file patterns the drift
-- check skips (code that only lists algorithm names). Both empty = one plan for the whole repo.
ALTER TABLE blueprints ADD COLUMN parts TEXT;
ALTER TABLE blueprints ADD COLUMN ignore_paths TEXT;
