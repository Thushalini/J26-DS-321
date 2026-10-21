import type { Prediction } from "@/lib/predict/types"

export const samplePrediction: Prediction = {
  source: "sample",
  enoughData: true,
  evaluatedAt: "2026-10-08",
  probability: 72,
  history: [58, 61, 60, 66, 72],
  factors: [
    {
      id: "model-performance",
      label: "Model performance vs target",
      contribution: 9,
      detail: "Validation score is within 2% of the target set in the blueprint.",
    },
    {
      id: "blueprint-drift",
      label: "Drift from blueprint",
      contribution: -7,
      detail: "Recent work has moved away from the agreed evaluation metric.",
    },
    {
      id: "timeline",
      label: "Timeline adherence",
      contribution: 6,
      detail: "Five of six planned milestones were completed on schedule.",
    },
    {
      id: "anomalies",
      label: "Open anomalies",
      contribution: -5,
      detail: "Three experiment anomalies are unresolved, one of high severity.",
    },
    {
      id: "data-quality",
      label: "Data quality",
      contribution: 4,
      detail: "Missing values and duplicates are below the agreed limits.",
    },
    {
      id: "mlops",
      label: "MLOps readiness",
      contribution: -3,
      detail: "Monitoring and drift detection are not yet in place.",
    },
    {
      id: "reproducibility",
      label: "Reproducibility",
      contribution: 2,
      detail: "Most experiments can be re-run from a fixed seed and config.",
    },
  ],
  forecast: [
    { week: 0, p10: 72, p25: 72, p50: 72, p75: 72, p90: 72 },
    { week: 1, p10: 68, p25: 71, p50: 73, p75: 75, p90: 77 },
    { week: 2, p10: 65, p25: 70, p50: 74, p75: 77, p90: 80 },
    { week: 3, p10: 62, p25: 69, p50: 75, p75: 79, p90: 83 },
    { week: 4, p10: 60, p25: 68, p50: 75, p75: 81, p90: 85 },
    { week: 5, p10: 58, p25: 67, p50: 76, p75: 82, p90: 87 },
    { week: 6, p10: 56, p25: 67, p50: 77, p75: 84, p90: 89 },
    { week: 7, p10: 54, p25: 66, p50: 77, p75: 85, p90: 90 },
    { week: 8, p10: 52, p25: 66, p50: 78, p75: 86, p90: 92 },
  ],
  readiness: {
    score: 62,
    areas: [
      {
        id: "versioning",
        label: "Model versioning",
        score: 78,
        reason: "Models and datasets are versioned; a few runs are untagged.",
      },
      {
        id: "pipeline",
        label: "Pipeline robustness",
        score: 64,
        reason: "The pipeline runs end to end but has no automated tests.",
      },
      {
        id: "documentation",
        label: "Documentation",
        score: 61,
        reason: "Setup is documented; modelling decisions are not.",
      },
      {
        id: "monitoring",
        label: "Monitoring and drift detection",
        score: 45,
        reason: "No monitoring of inputs or predictions after training.",
      },
    ],
  },
  recommendations: [
    {
      id: "realign-metric",
      title: "Re-align experiments with the agreed evaluation metric",
      reason: "Drift from the blueprint is the largest negative signal.",
      area: "Model evaluation",
      expectedGain: 5,
      support: { member: "Teammate B", strength: "model evaluation" },
    },
    {
      id: "resolve-anomaly",
      title: "Resolve the high-severity experiment anomaly",
      reason: "Unresolved anomalies keep lowering the probability each week.",
      area: "Experimentation",
      expectedGain: 3,
    },
    {
      id: "add-monitoring",
      title: "Add basic input and prediction monitoring",
      reason: "Monitoring is the weakest MLOps readiness area.",
      area: "Deployment and monitoring",
      expectedGain: 2,
      support: { member: "Teammate D", strength: "deployment and monitoring" },
    },
  ],
}
