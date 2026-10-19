"""The draft feature schema: one entry per prediction feature, with its source, scale and expected effect.

status "provisional" means the scale, weight or range has no confirmed source yet and must be
replaced before results are reported. Academic marks are deliberately absent.
Component 3 features follow Senthazhan's answers of 2026-10-18 (issue = severity MEDIUM or HIGH).
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    kind: str
    direction: int
    weight: float
    loading: float
    params: dict = field(default_factory=dict)
    reference: str | None = None
    status: str = "provisional"


RANGES_BY_KIND = {"unit": (0.0, 1.0), "score100": (0, 100), "count": (0, None)}
KINDS = set(RANGES_BY_KIND)

FEATURES = [
    FeatureSpec("novelty_score", "C1", "unit", +1, 0.3, 0.2),
    FeatureSpec("feasibility_score", "C1", "unit", +1, 0.8, 0.4),
    FeatureSpec("drift_score", "C2", "unit", -1, 1.0, 0.5),
    FeatureSpec("contribution_balance", "C2", "unit", +1, 0.7, 0.5),
    FeatureSpec("issue_count", "C3", "count", -1, 0.5, 0.5, {"mean": 2.5, "spread": 0.6}),
    FeatureSpec("issue_share", "C3", "unit", -1, 0.4, 0.5),
    FeatureSpec("max_severity", "C3", "unit", -1, 0.4, 0.4),
    FeatureSpec("mean_severity", "C3", "unit", -1, 0.3, 0.4),
    FeatureSpec("open_high_issues", "C3", "count", -1, 0.6, 0.5, {"mean": 0.8, "spread": 0.8}),
    FeatureSpec("mean_anomaly_score", "C3", "unit", -1, 0.2, 0.3),
    FeatureSpec("model_performance", "project_health", "unit", +1, 1.0, 0.6),
    FeatureSpec("data_quality", "project_health", "unit", +1, 0.7, 0.5),
    FeatureSpec("reproducibility", "project_health", "unit", +1, 0.6, 0.5),
    FeatureSpec("timeline_adherence", "project_health", "unit", +1, 1.0, 0.6),
    FeatureSpec("mlops_readiness_score", "mlops", "score100", +1, 0.5, 0.4),
]

AT_MOST = [
    ("open_high_issues", "issue_count"),
    ("mean_severity", "max_severity"),
]

ZERO_WHEN_ZERO = [
    ("issue_share", "issue_count"),
]


def feature_names() -> list[str]:
    return [spec.name for spec in FEATURES]


def get(name: str) -> FeatureSpec:
    return next(spec for spec in FEATURES if spec.name == name)
