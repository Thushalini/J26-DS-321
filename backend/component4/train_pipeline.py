"""Runs Component 4 end to end: generate data, validate it, train, calibrate, evaluate, explain, save.

Run from the backend folder:
    python -m component4.train_pipeline
"""
import json

import joblib

from .common import config
from .m2_prediction.data import split
from .m2_prediction.stability import seed_stability
from .m2_prediction.train import train_and_evaluate
from .m3_explainability.run import explain_model
from .m5_forecast.cone import cone_checks, forecast
from .m3_explainability.shap_explainer import local_explanation, shap_values
from .m6_improvement.failure_patterns import find_patterns, issues_from_c3_results
from .m6_improvement.recommendations import apply_changes, re_evaluate, recommend
from .synthetic import generator, validate

SUMMARY_METRICS = ["auc_roc", "brier", "ece", "macro_f1", "precision", "recall"]


def main():
    for folder in (config.DATA_DIR, config.MODELS_DIR, config.RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    frame = generator.generate()
    frame.to_csv(config.SYNTHETIC_CSV, index=False)

    checks = validate.run_checks(frame)
    validate.save_report(checks)
    blocking = [c["check"] for c in checks if not c["passed"] and c["check"] != "every_feature_sourced"]
    if blocking:
        raise SystemExit(f"Data validation failed: {blocking}. See {config.DATA_VALIDATION_JSON}")
    unsourced = next(c for c in checks if c["check"] == "every_feature_sourced")
    if not unsourced["passed"]:
        print(f"WARNING: {len(unsourced['detail'])} features are still provisional - results are not reportable yet.")

    report, model = train_and_evaluate(frame)
    report["data_provisional"] = not unsourced["passed"]
    config.METRICS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    x_train, x_test = split(frame)["train"][0], split(frame)["test"][0]
    joblib.dump({
        "model": model,
        "features": report["features"],
        "version": config.MODEL_VERSION,
        "background": x_train.sample(min(len(x_train), config.API["background_size"]), random_state=config.SEED),
        "medians": x_train.median().to_dict(),
        "demo_project": x_test.iloc[config.API["demo_project_index"]].to_dict(),
    }, config.MODEL_PATH)

    print(f"Calibration chosen: {report['calibration']['chosen']}  (validation Brier {report['calibration']['validation_brier']})")
    print(f"{'model':<24}" + "".join(f"{m:>10}" for m in SUMMARY_METRICS))
    for name, metrics in report["test_metrics"].items():
        print(f"{name:<24}" + "".join(f"{metrics[m]:>10.3f}" for m in SUMMARY_METRICS))
    print(f"Saved model to {config.MODEL_PATH} and metrics to {config.METRICS_JSON}")

    print("Tuned settings (5-fold search on the training split, scored by Brier):")
    for name, row in report["tuning"].items():
        print(f"  {name:<22}{row['best_params']}")
    print("Cross-validated on the training split (mean +/- std over 5 folds):")
    for name, scores in report["cross_validation"].items():
        print(f"  {name:<22}" + "  ".join(f"{m} {v['mean']:.3f}+/-{v['std']:.3f}" for m, v in scores.items()))

    stability = seed_stability(report["tuned_params"])
    config.SEED_STABILITY_JSON.write_text(json.dumps(stability, indent=2), encoding="utf-8")
    print(f"Seed stability over seeds {stability['seeds']} (test split, mean +/- std):")
    for name, scores in stability["summary"].items():
        print(f"  {name:<22}" + "  ".join(f"{m} {v['mean']:.3f}+/-{v['std']:.3f}" for m, v in scores.items()))

    parts = split(frame)
    explained = explain_model(model, parts["train"][0], parts["test"][0])
    config.EXPLAINABILITY_JSON.write_text(json.dumps(explained, indent=2), encoding="utf-8")
    print("Global importance (mean |SHAP|, probability points):")
    for row in explained["global_importance"]:
        print(f"  {row['feature']:<24}{row['mean_abs_points']:>8.2f}")
    failed = [r["feature"] for r in explained["direction_check"] + explained["sensitivity_analysis"] if not r["passed"]]
    print(f"Direction and sensitivity checks: {'all passed' if not failed else f'FAILED for {failed}'}")
    print(f"Saved explanations to {config.EXPLAINABILITY_JSON}")

    project = parts["test"][0].iloc[0]
    current = float(model.predict_proba(project.to_frame().T.astype(float))[:, 1][0])
    points = forecast(model, project)
    checks = cone_checks(points, current)
    config.FORECAST_EXAMPLE_JSON.write_text(json.dumps({"current_probability": round(current, 4), "points": points,
                                                        "checks": checks}, indent=2), encoding="utf-8")
    print(f"Forecast cone for one test project (now {current:.1%}): week {points[-1]['week']} median {points[-1]['p50']}%,"
          f" 80% range {points[-1]['p10']}-{points[-1]['p90']}%")
    print(f"Forecast checks: {'all passed' if all(c['passed'] for c in checks) else checks}")

    issues = issues_from_c3_results()
    patterns = find_patterns(issues)
    config.FAILURE_PATTERNS_JSON.write_text(json.dumps({k: v for k, v in patterns.items() if k != "assignments"},
                                                       indent=2), encoding="utf-8")
    print(f"Failure patterns in Component 3's issue history ({len(issues)} issues): {patterns['k']} patterns,"
          f" silhouette {patterns['silhouette']}")
    for p in patterns["patterns"]:
        print(f"  {' + '.join(p['causes']):<40}{p['issues']:>5} issues  mean severity {p['mean_severity']}")

    scenario = config.IMPROVEMENT_SCENARIO
    before = project.astype(float).copy()
    for name, value in scenario["starting_changes"].items():
        before[name] = value
    factors = local_explanation(shap_values(model, parts["train"][0], before.to_frame().T), 0)["factors"]
    suggestions = recommend(model, before, factors, scenario["recurring"])
    outcome = re_evaluate(model, before, apply_changes(before, config.RECOMMENDATIONS["actions"][suggestions[0]["id"]]["changes"]))         if suggestions else None
    config.IMPROVEMENT_SCENARIO_JSON.write_text(json.dumps({"scenario": scenario, "recommendations": suggestions,
                                                            "re_evaluation": outcome}, indent=2), encoding="utf-8")
    print("Level-6 scenario (low reproducibility, repeated issues) - recommendations:")
    for r in suggestions:
        print(f"  +{r['expected_gain_points']:>4} pts  {r['title']}  ({r['reason']})")
    if outcome:
        print(f"  After the top fix: {outcome['probability_before']:.1%} -> {outcome['probability_after']:.1%}")


if __name__ == "__main__":
    main()
