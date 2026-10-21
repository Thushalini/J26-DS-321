"""Builds one complete feature row for a project from the three adapters plus project-health values.

Anything a component could not provide yet stays missing; the prediction fills it with the training
median and says so, rather than failing or silently inventing a value.
"""
from ..synthetic.feature_spec import feature_names
from . import c1_adapter, c2_adapter, c3_adapter


def assemble(blueprint: dict | None = None, c2_summary: dict | None = None, c2_roles: dict | None = None,
             c3_runs: list[dict] | None = None, project_health: dict | None = None,
             readiness: dict | None = None) -> dict:
    features, context = {}, {}
    if blueprint is not None:
        part = c1_adapter.project_features(blueprint)
        features.update(part["features"])
        context["c1"] = part["context"]
    if c2_summary is not None:
        part = c2_adapter.project_features(c2_summary, c2_roles)
        features.update(part["features"])
        context["c2"] = part["context"]
    if c3_runs is not None:
        part = c3_adapter.project_features(c3_runs)
        features.update(part["features"])
        context["c3"] = part["context"]
    if readiness is not None:
        features["mlops_readiness_score"] = float(readiness["score"])
    features.update({k: v for k, v in (project_health or {}).items() if k in feature_names()})

    row = {name: features.get(name) for name in feature_names()}
    return {"features": row, "missing": [name for name, value in row.items() if value is None], "context": context}
