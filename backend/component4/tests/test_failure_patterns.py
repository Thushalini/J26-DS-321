import numpy as np

from component4.common import config
from component4.m6_improvement.failure_patterns import (find_patterns, issues_frame, issues_from_c3_results,
                                                        recurring_for_team)

CAUSES = config.FAILURE_PATTERNS["causes"]


def run(team, level, probs):
    return {"team_id": team, "severity_level": level, "severity_score": 0.8,
            "root_cause_probabilities": dict(zip(CAUSES, probs))}


def planted_runs():
    rng = np.random.default_rng(0)
    mixed = [0.9, 0.1, 0.85, 0.1, 0.1]
    single = [0.1, 0.9, 0.1, 0.1, 0.1]
    runs = [run("a", "HIGH", np.clip(np.array(mixed) + rng.normal(0, 0.03, 5), 0, 1)) for _ in range(12)]
    runs += [run("b", "MEDIUM", np.clip(np.array(single) + rng.normal(0, 0.03, 5), 0, 1)) for _ in range(12)]
    runs += [run("a", "LOW", single)]
    return runs


def test_only_medium_and_high_runs_become_issues():
    assert len(issues_frame(planted_runs())) == 24


def test_finds_a_mixed_cause_pattern():
    result = find_patterns(issues_frame(planted_runs()))
    labels = [sorted(p["causes"]) for p in result["patterns"]]
    assert sorted(["overfitting", "data_leakage"]) in labels and ["stalled"] in labels


def test_recurring_patterns_are_per_team():
    issues = issues_frame(planted_runs())
    result = find_patterns(issues)
    team_a = recurring_for_team(issues, result["assignments"], result["patterns"], "a")
    assert len(team_a) == 1 and sorted(team_a[0]["causes"]) == ["data_leakage", "overfitting"]


def test_reads_component3_saved_results():
    issues = issues_from_c3_results()
    assert len(issues) > 0 and set(CAUSES) <= set(issues.columns)


def test_too_few_issues_gives_no_patterns():
    assert find_patterns(issues_frame(planted_runs()[:1]))["patterns"] == []
