import pandas as pd

from component4.common import config
from component4.synthetic import generator, validate
from component4.synthetic.feature_spec import FEATURES

SMALL = 600


def checks_by_name(frame):
    return {c["check"]: c for c in validate.run_checks(frame)}


def test_same_seed_gives_identical_data():
    pd.testing.assert_frame_equal(generator.generate(SMALL, seed=7), generator.generate(SMALL, seed=7))


def test_different_seed_gives_different_data():
    assert not generator.generate(SMALL, seed=1).equals(generator.generate(SMALL, seed=2))


def test_full_size_data_passes_every_check_except_sources():
    failed = [name for name, c in checks_by_name(generator.generate()).items() if not c["passed"]]
    assert failed in ([], ["every_feature_sourced"])


def test_outcome_shares_match_config():
    shares = generator.generate(SMALL)[config.OUTCOME_COLUMN].value_counts(normalize=True)
    for outcome, expected in config.SYNTHETIC["outcome_shares"].items():
        assert abs(shares[outcome] - expected) <= config.DATA_CHECKS["share_tolerance"]


def test_checks_catch_planted_problems():
    frame = generator.generate(SMALL)
    frame.loc[0, "drift_score"] = 1.7
    frame.loc[1, config.ID_COLUMN] = frame.loc[2, config.ID_COLUMN]
    frame.loc[3, "data_quality"] = None
    frame.loc[4, config.OUTCOME_COLUMN] = "abandoned"
    checks = checks_by_name(frame)
    for name in ("values_in_range", "unique_project_ids", "no_missing_values", "known_outcomes"):
        assert not checks[name]["passed"], name


def test_checks_catch_a_reversed_relationship():
    frame = generator.generate(SMALL)
    frame["drift_score"] = 1 - frame["drift_score"]
    assert "drift_score" in checks_by_name(frame)["relationships_as_specified"]["detail"]


def test_no_feature_comes_from_academic_marks():
    assert not [spec.name for spec in FEATURES if spec.source in config.FORBIDDEN_FEATURE_SOURCES]
