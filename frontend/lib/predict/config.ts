import type { StatusLevel } from "@/lib/predict/types"

export const statusThresholds = {
  good: 70,
  warning: 50,
} as const

export const scoreMax = 100

export const statusLabels: Record<"probability" | "readiness", Record<StatusLevel, string>> = {
  probability: { good: "On track", warning: "At risk", critical: "Off track" },
  readiness: { good: "Ready", warning: "Needs work", critical: "Weak" },
}

export const gauge = {
  radius: 84,
  strokeWidth: 14,
} as const

export const sparkline = {
  width: 96,
  height: 28,
  padding: 6,
  dotRadius: 4,
} as const

export const forecastChart = {
  width: 320,
  height: 190,
  margin: { top: 10, right: 34, bottom: 24, left: 30 },
  yTicks: [0, 25, 50, 75, 100],
  xTickEvery: 2,
  dotRadius: 4,
  labelGap: 6,
} as const

export const predictCopy = {
  title: "Success prediction",
  evaluated: "Evaluated",
  sampleBadge: "Sample data",
  units: { points: "pts", week: "Week", now: "Now" },
  probability: {
    title: "Success probability",
    description: "Calibrated chance this project meets its goals.",
    delta: "since last evaluation",
    history: "Previous evaluations",
  },
  factors: {
    title: "What is driving it",
    description: "How much each signal moves the probability. Tap a bar for detail.",
    raises: "Raises probability",
    lowers: "Lowers probability",
  },
  forecast: {
    title: "Where it could go",
    description: "Simulated outcomes for the coming weeks if conditions vary.",
    median: "Median",
    inner: "Middle 50% of outcomes",
    outer: "80% of outcomes",
    range: "likely range",
  },
  readiness: {
    title: "MLOps readiness",
    description: "Rubric score across four areas.",
    outOf: "out of",
  },
  recommendations: {
    title: "What to do next",
    description: "Ordered by expected effect on the probability.",
    gain: "expected",
    support: "Suggested support",
    strength: "strongest in",
  },
} as const
