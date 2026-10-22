# Component 2 — Interface specification

**Component 2: DS Behavioural Modelling & Contribution Analytics** · owner: Thushalini Peranantham
Contract version **1.2** (`component2/api/contract.py`) · tests: `backend/tests/component2/test_c2_interfaces.py`

```
 INPUTS                                   COMPONENT 2                         OUTPUTS
 I1 GitHub repo (commits, notebooks) ─┐                                  ┌─> O1 /c2/teams/{id}/summary ──> C4 (prediction), C3
 I2 Team chat (app or CSV)          ──┼─> activities ─> stage labels ─>  ├─> O2 drift_results table   ──> C3 (drift_reader)
 I3 Blueprint (C1 / app)            ──┘      drift vs plan, weekly roles  ├─> O3 /c2/teams/{id}/roles  ──> C4 (contribution balance)
                                                                         └─> O4 /c2/teams/{id}/stages/daily ──> C3, C4
```

All timestamps are UTC text `YYYY-MM-DDTHH:MM:SSZ`. All people are anonymised member codes `M-xxxxxx`
(6 hex characters, `anonymise.make_code`); real names and emails never leave Component 2.
The seven stages are `data_cleaning, eda, feature_engineering, model_development, model_evaluation,
deployment_mlops, documentation` (`config.STAGES`).

## 1. Owners

| Interface | Direction | Producer (owner) | Consumer | Where it lives |
|---|---|---|---|---|
| I1 Repository activity | in | Team lead gives the repo URL; **C2 (Thushalini)** extracts it | C2 | `collect/extract_repos.py`, `buddy/sync.py` |
| I2 Team chat | in | **App layer (Thushalini, Bavatharani)** stores messages; C2 imports them | C2 | `chat_messages` table / `data/chat/<team_id>.csv` → `chat/load_chat.py` |
| I3 Project blueprint | in | **C1 (LakshiRajika)**; until C1 publishes per team, the team enters it in the app | C2 drift | `blueprints` table / `data/blueprints/<team_id>.json` |
| O1 Team summary | out | **C2 (Thushalini)** | C4 (Bavatharani), C3 (Senthazhan), frontend | `GET /c2/teams/{id}/summary`, `api/service.team_summary` |
| O2 Drift rows | out | **C2 (Thushalini)** | C3 (Senthazhan) | `drift_results` table → `component3/collector/drift_reader.py` |
| O3 Member roles | out | **C2 (Thushalini)** | C4 (Bavatharani), frontend | `GET /c2/teams/{id}/roles?window=` |
| O4 Daily stage counts | out | **C2 (Thushalini)** | C3, C4 (enough-work check) | `GET /c2/teams/{id}/stages/daily` |

The producer owns the format and must tell the consumer before changing it (see section 5).
The consumer owns what it does with a missing value (its fallback is listed in section 4).

## 2. Inputs (required data and format)

### I1 Repository activity and I2 Chat → one `activities` row each

| Field | Type | Required | Rule |
|---|---|---|---|
| `activity_id` | text | yes | unique, e.g. `commit:<team>:<sha>`, `chat:<team>:<msg_id>` |
| `team_id` | text | yes | same id across all components |
| `source` | text | yes | one of `commit`, `notebook_cell`, `code_change`, `chat` |
| `member_id` | text | no | `M-xxxxxx`; empty = could not be linked to a member |
| `ts` | text | yes | UTC `YYYY-MM-DDTHH:MM:SSZ`, not more than 1 day in the future |
| `text` | text | yes | not empty (commit message, cell source, code lines, chat message) |
| `meta` | JSON | no | e.g. `{"files": [...]}`, `{"reply_to": ...}` |

Chat CSV columns: `msg_id, timestamp, sender, message, reply_to` (one file per team, file name = `team_id`).
Checked by `contract.check_activity()`.

### I3 Blueprint (what the team planned)

C2 needs three decisions. Any of these shapes is accepted (`contract.clean_plan()`):

```json
{"target": "churn", "algorithm": "Random Forest", "metric": "F1"}                         // app blueprints table
{"target": {"name": "churn"}, "algorithm": {"name": "Random Forest"}, "metric": {"name": "F1"}}   // blueprint file
{"fields": {"target_variable": {"value": "churn", "status": "confirmed"},                // C1 Blueprint JSON
            "algorithm": {"value": "Random Forest", "status": "known"},
            "evaluation_metric": {"value": "F1", "status": "known"}}}
```

From C1, only fields with status `known` or `confirmed` count as the plan.

**Multi-model projects** (e.g. one model per component) add `parts`, and optionally `ignore`
(`contract.clean_blueprint()`; Blueprint tab → Parts, stored in `blueprints.parts` / `ignore_paths`):

```json
{"target": null, "algorithm": null, "metric": null,
 "parts": [{"name": "C2 stage classifier", "path": "backend/component2",
            "target": "stage", "algorithm": "logistic regression", "metric": "f1"},
           {"name": "C4 success prediction", "path": "backend/component4",
            "target": "success", "algorithm": "random forest", "metric": "roc auc"}],
 "ignore": ["backend/component2/classify/keyword_baseline.py", "backend/component2/drift/mentions.py", "*/tests/*"]}
```

- Up to 10 parts; a name is unique and has no `::`; `path` is a repo folder (deepest matching folder wins).
- Code changes and notebook cells go to the part their file is in. A commit goes to the part most of its
  files are in; a tie, chat, and files outside every part are compared with the project-level plan.
- `ignore`: a folder, a file, or a pattern (`*` also matches `/`). Use it for code that only *lists*
  algorithm / metric names. A commit that touched only ignored files is skipped.
- No `parts` = one plan for the whole repo, exactly as in 1.1.

## 3. Outputs (produced data and format)

### O1 `GET /c2/teams/{team_id}/summary?days=30` (the main contract)

```json
{
  "component": "C2",
  "version": "1.1",
  "team_id": "team-42",
  "generated_at": "2026-10-21T09:00:00Z",
  "window": {"start": "2026-09-21T09:00:00Z", "end": "2026-10-21T08:12:00Z"},
  "stage_breakdown": {"data_cleaning": 0.21, "eda": 0.18, "feature_engineering": 0.12,
                      "model_development": 0.30, "model_evaluation": 0.11,
                      "deployment_mlops": 0.03, "documentation": 0.05},
  "lagging_stages": ["deployment_mlops"],
  "drift": {
    "overall_severity": "moderate",
    "dimensions": [{"dimension": "algorithm", "planned": "random forest", "seen_top": "xgboost",
                    "drift_score": 0.45, "severity": "moderate", "explanation": "...",
                    "window_end": "2026-10-21T08:12:00Z", "part": null}],
    "parts": []
  },
  "members": [{"member_id": "M-3f9a2c", "stage_mix": {"eda": 0.6}, "chat": {}, "dominant_role": "eda",
               "role_trend": [{"week": "2026-W42", "dominant": "eda"}]}],
  "data_status": {"status": "ok", "last_activity": "2026-10-21T08:12:00Z",
                  "days_since_last_activity": 0, "stale_after_days": 14, "missing": []}
}
```

| Field | Meaning / allowed values |
|---|---|
| `stage_breakdown` | share (0–1) of recent commits + notebook cells per stage; exactly the 7 stages |
| `lagging_stages` | stages with < 5 % of recent work (provisional limit, agreed with C4) |
| `drift.dimensions[].severity` | `none` < 0.20 ≤ `minor` < 0.40 ≤ `moderate` < 0.60 ≤ `significant`; or `no_plan`, `no_evidence` (then `drift_score` is `null`) |
| `drift.overall_severity` | worst graded severity, or `unknown` if none is graded |
| `drift.dimensions[].part` | `null` = the project plan; otherwise the part this (worst) result comes from |
| `drift.parts` | multi-model projects: `[{"part": "<name>", "dimensions": [3 results like above]}]`; `[]` otherwise. The top-level `dimensions` are then the **worst** of the project and its parts, so C3 / C4 still get one result per decision |
| `data_status.status` | `ok`, `incomplete` (`missing` lists `stage_labels` / `drift` / `roles`), `stale` (no activity for > 14 days) |

`contract.check_summary(payload)` returns `[]` for a valid payload and a list of problems otherwise.

### O2 `drift_results` table (read directly by C3)

`team_id, window_end, dimension (target|algorithm|metric), planned, seen_top, drift_score (0–1 or NULL),
severity, explanation`, primary key `(team_id, window_end, dimension)`. A part's rows use
`dimension = "<part name>::<target|algorithm|metric>"`. C3 takes the latest window per dimension and uses
the largest score as its single `drift_severity_score` (so, for a multi-model project, the worst part).

### O3 `GET /c2/teams/{id}/roles?window=3d|7d|30d|all`

Per member: `code_items`, `chat_items`, `combined_mix`, `code_mix`, `chat_mix` (stage → share),
`main_roles`, `evidence` (`none` | `low` | `ok`), `last_active`; plus `chat_weight` (0.5).

### O4 `GET /c2/teams/{id}/stages/daily?days=30`

`[{"date": "2026-10-20", "data_cleaning": 3, ..., "documentation": 0, "other": 1}, ...]`

## 4. When data is missing, invalid or delayed

| Case | What Component 2 does | What the consumer does |
|---|---|---|
| **Missing** — unknown team / no activity | `404 No data for team` | C4 `_safe()` gets `None` → features filled with typical values and flagged; frontend shows "not enough data" |
| **Missing** — team has only chat, no repo synced | summary `404 has no repository activity yet` (was `null`) | same as above |
| **Missing** — no blueprint | research teams: plan inferred from first 40 % of the timeline; app teams: dimension reported `no_plan`, `drift_score: null` | C3: `source: no_score`, uses 0.0 and records it; C4: `drift_score` missing → typical value |
| **Missing** — blueprint field empty / C1 field not settled | that dimension `no_plan` | as above |
| **Missing** — drift / roles / labels not computed yet | summary still returned, `data_status.status = "incomplete"`, `missing` lists the parts | C4 treats absent drift as missing; lagging stages only trusted once there is enough labelled work |
| **Missing** — member with no activity | still listed, `evidence: "none"`, zero items | C4 lists them as inactive |
| **Invalid** — activity with bad / future timestamp, empty text, unknown source, no id | row skipped and reported with the reason (`check_activity`); the rest of the file loads | — |
| **Invalid** — blueprint file is not valid JSON | file kept as is (not overwritten), all dimensions `no_plan`, `blueprint: invalid_file`, warning printed | as "no blueprint" |
| **Invalid** — blueprint value not text / longer than 200 chars | treated as empty → `no_plan` | as "no blueprint" |
| **Invalid** — bad request parameters (`days=0`, `window=5d`) | `422` from FastAPI validation | fix the call |
| **Delayed** — no new activity for > 14 days (sync late or team quiet) | summary returned with `data_status.status = "stale"` and `days_since_last_activity` | C4/C3 may lower trust or show "data may be out of date" |
| **Delayed** — sync still running | app shows sync status (`teams` row); the last finished sync's results are served (drift and roles are only replaced in the final step) | read the last saved results |
| **Unavailable** — database down / table missing | API returns 500 | C3 `drift_reader`: `source: "unavailable: <error>"`, score `None`; C4 `_safe()` → `None` |

## 5. Working component-to-component connections

Both run today and are covered by `tests/component2/test_c2_interfaces.py`:

1. **C2 → C4**: `component4/m1_feature_integration/team_sources.py` calls `service.team_summary` and
   `service.member_roles`; `c2_adapter.project_features` turns them into `drift_score` (highest of
   target / algorithm / metric) and `contribution_balance` (1 − Gini). Used by C4's live team prediction.
2. **C2 → C3**: `component3/collector/drift_reader.read_drift(team_id)` reads `drift_results` and gives
   C3's models one `drift_severity_score`.

Demo:

```bash
cd backend
python -m pytest tests/component2 -v                         # contract + both connections + failure cases
uvicorn main:app --reload                                    # then open /docs
curl http://127.0.0.1:8000/c2/teams/<team_id>/summary
python -m component3.collector.drift_reader --team <team_id>
```

## 6. Changing the contract

- Adding a field: bump the minor version (1.0 → 1.1), tell C3 and C4. Old fields keep their meaning.
- Renaming / removing a field or changing a value's meaning: bump the major version (→ 2.0) and agree
  with C3 and C4 first.
- 1.1 (2026-10-21): added `data_status`; summary for a chat-only team is `404` instead of `null`.
- 1.2 (2026-10-21): multi-model projects: blueprint `parts` / `ignore`, `drift.parts`, `dimensions[].part`.
  New algorithm names `isolation forest`, `stacking`; `f1_macro` / `f1_weighted` count as F1.
