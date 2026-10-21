"""Rule-based synthetic project-outcome generator.

Each project gets a hidden overall-health value. Every feature follows it (by its loading) plus its
own noise, so features correlate the way real projects would. The outcome is then decided from the
observable features (weights and directions from the spec) plus noise, and cut into
success / delayed / failed at the configured shares.
"""
import numpy as np
import pandas as pd
from scipy.stats import beta, norm

from ..common import config
from .feature_spec import AT_MOST, FEATURES, ZERO_WHEN_ZERO, FeatureSpec


def _feature_values(spec: FeatureSpec, health: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    own = rng.standard_normal(health.size)
    latent = spec.direction * spec.loading * health + np.sqrt(1 - spec.loading**2) * own
    share = norm.cdf(latent)
    if "beta" in spec.params:
        share = beta.ppf(share, *spec.params["beta"])
    if spec.kind == "unit":
        return np.round(share, 4)
    if spec.kind == "score100":
        return np.round(100 * share).astype(int)
    if spec.kind == "count":
        rate = spec.params["mean"] * np.exp(spec.params["spread"] * latent)
        return rng.poisson(rate)
    raise ValueError(f"Unknown feature kind: {spec.kind}")


def _apply_constraints(frame: pd.DataFrame) -> pd.DataFrame:
    for smaller, larger in AT_MOST:
        frame[smaller] = np.minimum(frame[smaller], frame[larger])
    for dependent, base in ZERO_WHEN_ZERO:
        frame.loc[frame[base] == 0, dependent] = 0
    return frame


def _standardise(values: np.ndarray) -> np.ndarray:
    spread = values.std()
    return (values - values.mean()) / spread if spread else values * 0.0


def _outcomes(frame: pd.DataFrame, rng: np.random.Generator, settings: dict) -> pd.Series:
    score = sum(spec.weight * spec.direction * _standardise(frame[spec.name].to_numpy(float)) for spec in FEATURES)
    score = score + rng.logistic(scale=settings["outcome_noise_scale"] * np.std(score), size=len(frame))

    shares = settings["outcome_shares"]
    failed_cut = np.quantile(score, shares["failed"])
    delayed_cut = np.quantile(score, shares["failed"] + shares["delayed"])
    labels = np.where(score < failed_cut, "failed", np.where(score < delayed_cut, "delayed", "success"))
    return pd.Series(labels, index=frame.index, name=config.OUTCOME_COLUMN)


def generate(n_projects: int | None = None, seed: int = config.SEED, settings: dict | None = None) -> pd.DataFrame:
    settings = {**config.SYNTHETIC, **(settings or {})}
    n_projects = n_projects or settings["n_projects"]
    rng = np.random.default_rng(seed)
    health = rng.standard_normal(n_projects)

    frame = pd.DataFrame({config.ID_COLUMN: [f"p{index:05d}" for index in range(1, n_projects + 1)]})
    for spec in FEATURES:
        frame[spec.name] = _feature_values(spec, health, rng)
    frame = _apply_constraints(frame)
    frame[config.OUTCOME_COLUMN] = _outcomes(frame, rng, settings)
    return frame


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = generate()
    frame.to_csv(config.SYNTHETIC_CSV, index=False)
    print(f"Saved {len(frame)} projects to {config.SYNTHETIC_CSV}")
    print(frame[config.OUTCOME_COLUMN].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
