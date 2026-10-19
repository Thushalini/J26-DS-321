"""Runs Component 4 end to end: generate data, validate it, train, calibrate, evaluate, explain, save.

Run from the backend folder:
    python -m component4.train_pipeline
"""
import json

import joblib

from .common import config
from .m2_prediction.data import split
from .m2_prediction.train import train_and_evaluate
from .m3_explainability.run import explain_model
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
    joblib.dump({"model": model, "features": report["features"], "version": config.MODEL_VERSION}, config.MODEL_PATH)

    print(f"Calibration chosen: {report['calibration']['chosen']}  (validation Brier {report['calibration']['validation_brier']})")
    print(f"{'model':<24}" + "".join(f"{m:>10}" for m in SUMMARY_METRICS))
    for name, metrics in report["test_metrics"].items():
        print(f"{name:<24}" + "".join(f"{metrics[m]:>10.3f}" for m in SUMMARY_METRICS))
    print(f"Saved model to {config.MODEL_PATH} and metrics to {config.METRICS_JSON}")

    parts = split(frame)
    explained = explain_model(model, parts["train"][0], parts["test"][0])
    config.EXPLAINABILITY_JSON.write_text(json.dumps(explained, indent=2), encoding="utf-8")
    print("Global importance (mean |SHAP|, probability points):")
    for row in explained["global_importance"]:
        print(f"  {row['feature']:<24}{row['mean_abs_points']:>8.2f}")
    failed = [r["feature"] for r in explained["direction_check"] + explained["sensitivity_analysis"] if not r["passed"]]
    print(f"Direction and sensitivity checks: {'all passed' if not failed else f'FAILED for {failed}'}")
    print(f"Saved explanations to {config.EXPLAINABILITY_JSON}")


if __name__ == "__main__":
    main()
