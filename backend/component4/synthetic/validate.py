"""Validation level 1: checks a project dataset before any model sees it."""
import json

import pandas as pd

from ..common import config
from .feature_spec import AT_MOST, FEATURES, KINDS, RANGES_BY_KIND, ZERO_WHEN_ZERO


def _check(name: str, passed: bool, detail) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def _out_of_range(frame: pd.DataFrame) -> dict:
    problems = {}
    for spec in FEATURES:
        low, high = RANGES_BY_KIND[spec.kind]
        values = frame[spec.name]
        bad = (values < low) | ((values > high) if high is not None else False)
        if bad.any():
            problems[spec.name] = int(bad.sum())
    return problems


def _broken_constraints(frame: pd.DataFrame) -> dict:
    problems = {}
    for smaller, larger in AT_MOST:
        bad = int((frame[smaller] > frame[larger]).sum())
        if bad:
            problems[f"{smaller} <= {larger}"] = bad
    for dependent, base in ZERO_WHEN_ZERO:
        bad = int(((frame[base] == 0) & (frame[dependent] != 0)).sum())
        if bad:
            problems[f"{dependent} = 0 when {base} = 0"] = bad
    return problems


def _wrong_directions(frame: pd.DataFrame) -> dict:
    success = frame[config.OUTCOME_COLUMN].isin(config.POSITIVE_OUTCOMES).astype(int)
    minimum = config.DATA_CHECKS["min_abs_correlation"]
    problems = {}
    for spec in FEATURES:
        rho = frame[spec.name].corr(success, method="spearman")
        if pd.isna(rho) or rho * spec.direction < minimum:
            problems[spec.name] = None if pd.isna(rho) else round(float(rho), 3)
    return problems


def _share_gaps(frame: pd.DataFrame, expected: dict) -> dict:
    observed = frame[config.OUTCOME_COLUMN].value_counts(normalize=True)
    tolerance = config.DATA_CHECKS["share_tolerance"]
    return {
        outcome: round(float(observed.get(outcome, 0.0)), 3)
        for outcome, share in expected.items()
        if abs(observed.get(outcome, 0.0) - share) > tolerance
    }


def run_checks(frame: pd.DataFrame, expected_shares: dict | None = None) -> list[dict]:
    expected_shares = expected_shares or config.SYNTHETIC["outcome_shares"]
    columns = [config.ID_COLUMN, *(spec.name for spec in FEATURES), config.OUTCOME_COLUMN]
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        return [_check("required_columns", False, missing_columns)]

    feature_columns = [spec.name for spec in FEATURES]
    missing = frame[columns].isna().sum()
    unknown_outcomes = sorted(set(frame[config.OUTCOME_COLUMN]) - set(config.OUTCOMES))
    out_of_range = _out_of_range(frame)
    broken = _broken_constraints(frame)
    wrong_directions = _wrong_directions(frame)
    share_gaps = _share_gaps(frame, expected_shares)
    forbidden = [spec.name for spec in FEATURES if spec.source in config.FORBIDDEN_FEATURE_SOURCES]
    bad_kinds = [spec.name for spec in FEATURES if spec.kind not in KINDS]
    provisional = [spec.name for spec in FEATURES if spec.status != "confirmed" or not spec.reference]

    return [
        _check("required_columns", True, []),
        _check("no_missing_values", missing.sum() == 0, missing[missing > 0].to_dict()),
        _check("known_outcomes", not unknown_outcomes, unknown_outcomes),
        _check("values_in_range", not out_of_range, out_of_range),
        _check("features_consistent", not broken, broken),
        _check("unique_project_ids", frame[config.ID_COLUMN].is_unique, int(frame[config.ID_COLUMN].duplicated().sum())),
        _check("no_duplicate_rows", not frame[feature_columns].duplicated().any(), int(frame[feature_columns].duplicated().sum())),
        _check("outcome_shares_as_configured", not share_gaps, share_gaps),
        _check("relationships_as_specified", not wrong_directions, wrong_directions),
        _check("no_marks_in_features", not forbidden, forbidden),
        _check("known_feature_kinds", not bad_kinds, bad_kinds),
        _check("every_feature_sourced", not provisional, provisional),
    ]


def save_report(results: list[dict]) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_VALIDATION_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    frame = pd.read_csv(config.SYNTHETIC_CSV)
    results = run_checks(frame)
    save_report(results)
    for result in results:
        print(f"{'PASS' if result['passed'] else 'FAIL'}  {result['check']}  {result['detail'] or ''}")


if __name__ == "__main__":
    main()
