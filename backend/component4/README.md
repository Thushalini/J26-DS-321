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

Outputs: `data/synthetic_projects.csv`, `models/success_model.joblib`, and in `results/`: `data_validation.json`, `metrics.json` (incl. tuning and cross-validation), `seed_stability.json`, `explainability.json`, `forecast_example.json`, `failure_patterns.json`, `improvement_scenario.json`. The full run takes about 10 minutes.

## Status

- **Synthetic data (O2):** generator and level-1 checks work. Every feature in `synthetic/feature_spec.py` is still **provisional** — scales, weights and ranges have no confirmed source, so results are not reportable yet. The pipeline prints a warning until every feature has a reference.
- **Prediction (O3):** baselines (rule-based, LogReg, RF, GB), stacking ensemble, Platt vs isotonic calibration chosen on validation Brier, and test metrics (precision, recall, macro-F1, AUC-ROC, Brier, ECE, calibration curve). Base models tuned by grid search (Brier), 5-fold cross-validation and a 5-seed stability check.
- **Explainability (O4):** permutation SHAP on the calibrated probability (contributions in probability points that add up to the prediction), global importance, per-project explanations, direction check and one-step sensitivity analysis.
- **Feature integration (O1):** adapters for Component 1 (novelty; feasibility checks pass/warning/fail -> one 0-1 score), Component 2 (drift = worst of target/algorithm/metric with its driver; contribution balance = 1 - Gini of member activity) and Component 3 (issue features), and `assemble.py`, which builds one full feature row and lists what is still missing.
- **Forecast cone (O5):** Monte Carlo simulation of the project week by week (score drift, new issues, resolved issues) scored by the model; percentiles per week in the shape the Predict screen uses; optional improvement trend; level-5 checks (starts at the current prediction, ordered, widens).
- **Improvement engine (O6):** recurring failure patterns from Component 3's issue history (k-means on root-cause probabilities, k by silhouette); recommendations triggered by negative SHAP signals or recurring patterns, ranked by expected gain from re-scoring; before/after re-evaluation; level-6 scenario.
- **API:** `/c4/health`, `/c4/features`, `POST /c4/predict` (features -> probability, SHAP factors, forecast cone, recommendations; missing features are filled with the training median and listed) and `/c4/demo` (a held-out synthetic project). Registered in `backend/main.py`; try it at http://127.0.0.1:8000/docs.
- **Not started:** readiness rubric (needs the rubric wording), competency branch (needs the marks and module outlines), connecting the Predict screen to the API.
