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
  collector/               reads real data to feed predict_live.py
    git_collector.py         commits per day, commit gap, notebook edits from a Git repo
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

## Results (test split, 210 runs)

| Check | Result | Target |
|---|---|---|
| Stage 1 AUC-ROC | 0.9998 | > 0.85 |
| Stage 2 macro F1 | 0.9568 | > 0.80 |
| Stage 2 Brier score | 0.0157 | < 0.15 |
| Stage 2 Hamming loss | 0.0114 | lower than baseline |
| Macro F1 gain vs Baseline A (generic rules) | +21.9% | >= 10% (met) |
| Macro F1 gain vs Baseline B (best single rule per cause) | +5.6% | >= 10% (not met) |
| HIGH-severity cases correctly triggered | 133 / 133 | > 85% |
| KS test (synthetic vs real loss values) | p = 0.136 | > 0.05 |

## Limitations

- Synthetic, cleanly separated data: scores are likely higher than on real experiments.
- The KS test uses only 6 real values from a different task, so it is a basic realism check.
- Removing the anomaly score or the drift score from the Stage 2 inputs did not change its results on this data.
- `drift_severity_score` (from Component 2) is a simulated feature that separates classes very easily.
- `predict_live.py` scores a run that is handed to it; the Git part of the collector (FR1) is built; reading MLflow and W&B is still to be done.

## Status

Stages 1-3, the baselines, the ablation study and the single-run prototype (`predict_live.py`) are finished. The data collector (FR1) and the final JSON contract with Component 4 / the frontend are still to be done.
