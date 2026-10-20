# Component 3 - Real-Time Collaboration Intelligence & Adaptive Intervention

Watches a data science team's experiment runs and answers three questions:

1. **Is this run healthy or anomalous?** (Stage 1 - Isolation Forest, anomaly score 0-1)
2. **If anomalous, why?** (Stage 2 - calibrated Gradient Boosting, a probability for each of 5 root causes: overfitting, stalled, data leakage, resource bottleneck, premature evaluation)
3. **How urgent is it and what should the team do?** (Stage 3 - `severity = 0.6 * anomaly + 0.4 * top cause probability`, level LOW / MEDIUM / HIGH, template advice)

The trained models are compared with rule-based baselines in an ablation study.

> The data is **synthetic** (1,400 labelled runs made by `scripts/generate_dataset.py`), so the near-perfect scores below show the pipeline works on clean simulated failures, not that it will perform as well on real experiments.

## Folder layout

```
component3/
  train_pipeline.py        runs everything and saves models + results
  predict_live.py          scores ONE new run and returns the result as JSON
  common/                  shared parts
    config.py                paths, seed, names of signals and causes
    splits.py                train / val / test split by run (70/15/15)
    features.py              epoch rows -> one row of 31 features per run
  stage1_anomaly/          Stage 1
    anomaly_model.py         Isolation Forest + 0-1 anomaly score
  stage2_root_cause/       Stage 2
    root_cause_model.py      calibrated Gradient Boosting (5 causes)
  stage3_severity/         Stage 3
    severity.py              severity score, level, advice templates
  api/                     web API and database storage for the Alerts page
  collector/               reads real data to feed predict_live.py
    git_collector.py         commits per day, commit gap, notebook edits from a Git repo
    notebook_logs.py         finds training runs (epoch logs) in a team's notebooks
    assemble.py              joins tracker epochs + Git + drift score into one run
  evaluation/              proving the ML beats simple rules
    baseline.py              rule-based baselines A and B
    ablation.py              baseline vs ML comparison + feature ablations
  data/                    the datasets, splits.csv, real reference sample
    samples/                 two example run inputs for predict_live.py
  models/                  saved models (created by train_pipeline)
  results/                 metrics and per-run outputs (created by train_pipeline)
  scripts/                 generate_dataset.py, ks_validation.py
```

## How to run

From the **repository root** (the folder that contains `backend/`), with the packages from `backend/requirements.txt` installed:

```
python -m backend.component3.train_pipeline
```

This takes about a minute and prints the results of every stage. It saves the models to `models/` and the outputs to `results/` (`metrics.json` has the summary).

Check the realism of the synthetic data (KS test, expected p = 0.136):

```
python backend/component3/scripts/ks_validation.py
```

Do **not** re-run `scripts/generate_dataset.py` unless you want to rebuild the data: it overwrites the CSVs in `data/`.

## Score a single run (live prototype)

After training (models must exist in `models/`), score one run from its numbers:

```
python -m backend.component3.predict_live --run backend/component3/data/samples/sample_run_overfitting.json
python -m backend.component3.predict_live --demo run_00501
```

Add `--log` to also append the result to `results/issue_history.jsonl` (the per-run issue history for Component 4).
From Python: `from backend.component3.predict_live import predict_run`.

**Input:** the 5 signals for every epoch (`train_accuracy`, `val_accuracy`, `loss`, `gpu_utilization_pct`, `commits_that_day`) plus 4 context values (`drift_severity_score` from Component 2, `hyperparam_tuning_logged`, `commit_gap_days`, `notebook_edit_freq_per_day`). See the top of `predict_live.py` for the exact format.
**Output (JSON):** anomaly score, a probability for each cause, severity score and level, a short `title` and one-line `evidence` (for an alert card), recommended cause and advice. The field names are a first draft (`schema_version` 0.1-draft) and can be renamed to match Component 4 and the frontend.
A call takes well under 1 second (requirement: under 5 seconds), and the live path reproduces the batch results (same severity level for all 210 test runs).

## Collect data from Git

`collector/git_collector.py` reads a repository's history and gives the commit-related inputs for a time window (normally the start and end of one run):

```
python -m backend.component3.collector.git_collector --repo . --since 2026-10-01 --until 2026-10-14 --utc-offset 5.5
```

It returns commits per day, the days since the last commit, and notebook edits per day. It uses the `git` command, so no extra Python package is needed. Merge commits are skipped, and days are counted in UTC unless `--utc-offset` is given.
`collector/assemble.py` then joins these with the epochs from an experiment tracker (each epoch needs a `timestamp`) and the drift score from Component 2 into the run that `predict_run()` scores.

Limit: Git records commits, not every save of a notebook, so "notebook edits per day" is a lower estimate than the training data used.

## Alerts API (for the web app)

The website's Alerts page loads each team's alerts from the backend (`backend/component3/api/`). Add to `backend/main.py`:

```
from component3.api.router import router as c3_router
app.include_router(c3_router)
```

| Address | Who may use it | What it does |
|---|---|---|
| `GET /c3/teams/{team_id}/alerts` | any signed-in user who can see the team | the team's alerts: open, resolved and healthy runs |
| `POST /c3/teams/{team_id}/alerts/{run_id}/resolve` | the team's lead or members (not research teams) | mark an alert resolved, or reopen it |
| `POST /c3/teams/{team_id}/runs` | collectors and scripts with the `X-C3-Key` header (= `C3_API_KEY` in `backend/.env`) | score a run and save its alert |

The routes use the same login and access checks as the Buddy.ds routes (`team_access`, `require`, `require_live`).

Alerts are stored in the table `component3_alerts` (one row per team and run), using the same plain column types and the same database wrapper as Component 2, so it works on PostgreSQL and on the local SQLite file. To fill a team with sample alerts for a demo, run from the `backend` folder: `python -m component3.api.seed_alerts --find <team name>` and then `python -m component3.api.seed_alerts --team <team_id>`.

## Find training runs in a notebook

`collector/notebook_logs.py` reads the training logs that Jupyter saves inside an `.ipynb` file (Keras 2 and 3, PyTorch-style prints and similar `loss` / `accuracy` / `val_loss` / `val_accuracy` logs) and turns each training loop into a run:

```
python -m backend.component3.collector.notebook_logs path/to/notebook.ipynb --score
```

A loop is used only when it has at least 3 epochs and shows the loss and both training and validation accuracy. GPU use is not in a notebook, so a neutral value is assumed and the output lists what was assumed.

## Results

Version 1 learned from **simulated** runs only and scored them almost perfectly, but on real training runs it raised an alert for **every** healthy run (100% false alarms). Version 2, the default, also learns from real training runs (`data/real_runs/`, 1,000 real runs on 8 tasks and 3 model types, made by `scripts/make_real_runs.py`). `python -m backend.component3.train_pipeline --simulated-only` recreates version 1.

| Test set | Healthy runs wrongly alerted | Failing runs alerted | Right cause when alerted | Macro F1 |
|---|---|---|---|---|
| Simulated (210 runs) | 0 of 75 | 134 of 135 | 134 of 134 | 0.952 |
| Real (150 runs, the kinds of tasks it trained on) | 2 of 60 | 70 of 90 | 70 of 70 | 0.724 |

Stage 1 AUC-ROC: 0.992 simulated, 0.882 real. Brier score: 0.014 simulated, 0.023 real.

**Unseen tasks and models** (`evaluation/generalization.py`; the pipeline is retrained without those runs and tested on them):

| Held out | Healthy runs wrongly alerted | Failures alerted | Best single rule per cause (healthy wrongly alerted / failures alerted) |
|---|---|---|---|
| Random 15% of real runs | 3% | 74% | 35% / 90% |
| Whole tasks (breast cancer, wine, iris) | 12% | 79% | 48% / 90% |
| A whole model type (SGD logistic regression) | 15% | 72% | 37% / 84% |

**Against the rule baselines.** The trained pipeline gives far fewer false alarms than the rules, and fewer errors per label (Hamming loss), but on real runs the tuned single rules catch more failures, mostly **data leakage**, and so get a higher macro F1 (0.85 against 0.72). Data leakage is mostly missed on real runs: a leaking run looks like a healthy run on an easy task.

On the simulated test set the macro F1 gain over the generic rules is +21.3% (target of 10% met) and over the tuned rules +7.7% (not met).

## Limitations

- The real runs are lab-made: standard and generated tasks trained with scikit-learn, not a team's own experiments. GPU use, commits, commit gap, notebook edits and the drift score are filled with varied random values, so only the training curves and the number of epochs are real.
- There are **no real resource-bottleneck runs** (no GPU to measure), so that cause is covered by simulated runs only.
- **Data leakage is mostly missed on real runs**, and 12 to 15% of healthy runs on unseen tasks or models still raise an alert.
- Test sets are small (150 real runs in the standard split, 274 to 374 in the held-out tests).
- The first stage depends on the absolute shape of the curves; features that do not depend on the task did worse in a trial.
- `drift_severity_score` (from Component 2) and the anomaly score add little to the root-cause stage; removing them changed the results only slightly.
- Real teams' own experiments have not been tested yet. The notebook log reader (below) is the way in.
- Automatic collection from MLflow and Weights & Biases (FR1) is not built yet; the Git collector, the drift reader and the notebook log reader are.

## Status

Built and merged: the three stages, the baselines and ablation, single-run scoring, the Git collector, the drift reader, per-team alerts in the database and the Alerts page. Version 2 retrains on real runs. Still to do: MLflow / Weights & Biases readers, scoring new runs automatically, and testing on real teams' experiments.
