# Component 4 — Predictive Success Evaluation & Continuous MLOps Readiness

Final evaluation layer. Takes the structured outputs of Components 1–3 plus project-health indicators and runs Evaluate → Explain → Improve → Re-evaluate.

## Folder layout

```
component4/
  common/                   config.py — paths, seed and every setting (nothing hard-coded elsewhere)
  m1_feature_integration/   O1 — one feature row per project from C1–C3 + project health
  m2_prediction/            O3 — calibrated stacking ensemble (RF + GB + LogReg)
  m3_explainability/        O4 — SHAP, direction checks, sensitivity analysis
  m4_readiness/             O5 — MLOps readiness rubric
  m5_forecast/              O5 — Monte Carlo forecast cone
  m6_improvement/           O6 — improvement engine and re-evaluation
    competency/               the only place academic marks are used
  synthetic/                O2 — synthetic dataset generator and data validation
  api/                      FastAPI router
  data/                     datasets (data/marks/ is git-ignored and never committed)
  models/                   saved models
  results/                  metrics and outputs
  tests/                    pytest tests
```

**Academic marks are never features of the prediction model.** They are read only by `m6_improvement/competency/`, after a weak area has been identified.

## How to run

From the `backend/` folder, with the packages from `requirements.txt` installed.

Generate data, validate it, train, calibrate and evaluate (about a minute):

```bash
python -m component4.train_pipeline
```

Run the tests:

```bash
python -m pytest component4/tests
```

Outputs: `data/synthetic_projects.csv`, `results/data_validation.json`, `results/metrics.json`, `results/explainability.json`, `models/success_model.joblib`. The full run takes about 3 minutes, most of it SHAP.

## Status

- **Synthetic data (O2):** generator and level-1 checks work. Every feature in `synthetic/feature_spec.py` is still **provisional** — scales, weights and ranges have no confirmed source, so results are not reportable yet. The pipeline prints a warning until every feature has a reference.
- **Prediction (O3):** baselines (rule-based, LogReg, RF, GB), stacking ensemble, Platt vs isotonic calibration chosen on validation Brier, and test metrics (precision, recall, macro-F1, AUC-ROC, Brier, ECE, calibration curve).
- **Explainability (O4):** permutation SHAP on the calibrated probability (contributions in probability points that add up to the prediction), global importance, per-project explanations, direction check and one-step sensitivity analysis.
- **Feature integration (O1):** Component 3 adapter (`m1_feature_integration/c3_adapter.py`) turns per-run results into project features, as agreed with Component 3. Components 1 and 2 adapters wait for their answers.
- **Not started:** readiness rubric, forecast cone, improvement engine, API.
