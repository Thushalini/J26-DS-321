"""The feature schema: one entry per prediction feature, with its source, scale, expected effect and evidence.

weight follows evidence tiers: 1.0 = part of the definition of success, 0.5 = significant in empirical studies
(split across a construct's indicators), 0.2-0.3 = weak or mixed evidence. direction 0 = neutral: no effect on
success (novelty, the student's decision, because the evidence points both ways). weight_sources back the direction and
weight; range_sources back the spread ("beta" in params), otherwise the spread is a placeholder. status turns
"confirmed" only after the student has checked every source listed (see sources.py). Marks are deliberately absent.
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
    weight_sources: tuple = ()
    range_sources: tuple = ()
    status: str = "proposed"


RANGES_BY_KIND = {"unit": (0.0, 1.0), "score100": (0, 100), "count": (0, None)}
KINDS = set(RANGES_BY_KIND)

EXPERIMENT_HEALTH = ("forouzeshnejad2025", "prause2026")

FEATURES = [
    FeatureSpec("novelty_score", "C1", "unit", 0, 0.0, 0.0, weight_sources=("wang2017",)),
    FeatureSpec("feasibility_score", "C1", "unit", +1, 0.4, 0.4, weight_sources=("prause2026", "rand2024")),
    FeatureSpec("drift_score", "C2", "unit", -1, 0.4, 0.5, weight_sources=("prause2026", "zowghi2002")),
    FeatureSpec("contribution_balance", "C2", "unit", +1, 0.2, 0.5, {"beta": (2.0, 2.4)},
                weight_sources=("kopec2023", "martinez2021"), range_sources=("kopec2023",)),
    FeatureSpec("issue_count", "C3", "count", -1, 0.05, 0.5, {"mean": 2.5, "spread": 0.6}, weight_sources=EXPERIMENT_HEALTH),
    FeatureSpec("issue_share", "C3", "unit", -1, 0.12, 0.5, weight_sources=EXPERIMENT_HEALTH),
    FeatureSpec("max_severity", "C3", "unit", -1, 0.08, 0.4, weight_sources=EXPERIMENT_HEALTH),
    FeatureSpec("mean_severity", "C3", "unit", -1, 0.06, 0.4, weight_sources=EXPERIMENT_HEALTH),
    FeatureSpec("open_high_issues", "C3", "count", -1, 0.15, 0.5, {"mean": 0.8, "spread": 0.8}, weight_sources=EXPERIMENT_HEALTH),
    FeatureSpec("mean_anomaly_score", "C3", "unit", -1, 0.04, 0.3, weight_sources=EXPERIMENT_HEALTH),
    FeatureSpec("model_performance", "project_health", "unit", +1, 1.0, 0.6, weight_sources=("prause2026",)),
    FeatureSpec("data_quality", "project_health", "unit", +1, 0.5, 0.5, weight_sources=("rand2024", "sambasivan2021")),
    FeatureSpec("reproducibility", "project_health", "unit", +1, 0.3, 0.5, {"beta": (2.0, 6.0)},
                weight_sources=("gundersen2018", "serban2020"), range_sources=("gundersen2018",)),
    FeatureSpec("timeline_adherence", "project_health", "unit", +1, 1.0, 0.6,
                weight_sources=("chaos2020", "nenni2025"), range_sources=("chaos2020",)),
    FeatureSpec("mlops_readiness_score", "mlops", "score100", +1, 0.5, 0.4, {"beta": (2.0, 4.0)},
                weight_sources=("prause2026", "serban2020"), range_sources=("breck2017", "prause2026")),
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
