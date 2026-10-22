import type { StatusLevel } from "@/lib/predict/types"

export const statusThresholds = {
  good: 70,
  warning: 50,
} as const

export const scoreMax = 100

export const pointsDecimals = 1

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
    range: "Likely {low}–{high}%. The range narrows as more signals are measured.",
    rangeMinWidth: 1,
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
    description: "Draft rubric across four areas. Buddy reads the group's key repo files: README, tests, CI, requirements and Dockerfile.",
    namesOnly: "Draft rubric across four areas, from file names only: GitHub couldn't be read, so file contents weren't checked.",
    outOf: "out of",
  },
  recommendations: {
    title: "What to do next",
    description: "Ordered by expected effect on the probability.",
    gain: "expected",
    support: "Suggested support",
    strength: "strongest in",
  },
  live: {
    loading: "Loading the prediction…",
    noTeam: "Open a group first.",
    signals: "Based on {n} of {total} signals from this team. The rest use typical values until every component sends its data.",
    notEnoughTitle: "Not enough data yet",
    notEnough: "Buddy gives a number once at least {min} of {total} signals come from this group. It has {n} so far. Sync the GitHub repository in Behavior and add experiment runs in Alerts.",
    notMeasured: "Not measured yet, so not used in the reasons: {items}.",
    changed: "Since {date}:",
    readinessNone: "Sync the group's GitHub repository in Behavior to score this.",
    decimals: 2,
    retry: "Try again",
  },
  summary: {
    title: "Summary",
    description: "Success outlook, likely direction and MLOps readiness for this group.",
  },
  pulse: {
    title: "Pulse",
    more: "See Predict for the reasons and what to do next.",
  },
  timeline: {
    title: "Timeline",
    description: "The shared final-year schedule plus the blueprint's own deadline and milestones. A milestone counts as met when Behavior and experiments show work in the stages expected by then.",
    upcoming: "upcoming",
    met: "met",
    missed: "behind",
    noStages: "no stages to check",
    planning: "planning step",
    scheduleTag: "shared schedule",
    basis: "Shared schedule: {schedule} ({status}). Dates are counted from {start}, {from}.",
    basisNoSchedule: "Dates are counted from {start}, {from}.",
    startFrom: {
      repo: "when the repo was created",
      first_commit: "the first commit",
      group: "when the group was made in Buddy",
    },
    scheduleStatus: {
      sample: "entered by hand for now",
    } as Record<string, string>,
  },
  milestones: {
    title: "Milestone results",
    description: "After each milestone (e.g. PP1, a viva), record how it went. Buddy compares this with its earlier predictions to check itself.",
    empty: "No results recorded yet.",
    name: "Milestone",
    namePlaceholder: "e.g. Progress presentation 1",
    date: "Date",
    result: "How it went",
    save: "Save result",
    saving: "Saving…",
    remove: "Remove",
    readOnly: "Only the group's members can record results.",
    options: { passed_well: "Passed well", passed_with_issues: "Passed with issues", needs_rework: "Needs rework" },
  },
  helpers: {
    title: "Who could help",
    description: "Add each member's Student Performance Profile (PDF). Buddy finds who is strongest in the modules behind each stage that needs help.",
    privacy: "Sheets are saved for this group so they only need adding once. Nobody's marks are shown, only the suggested person.",
    memberFile: "Add or replace score sheet (PDF)",
    saved: "Sheet saved",
    notSaved: "No sheet yet",
    uploading: "Saving…",
    remove: "Remove",
    stagesTitle: "Stages to find help for",
    stagesDetected: "Ticked automatically from this group's data. You can add or untick stages.",
    stagesNone: "Nothing detected from the data yet. Sync the repository in Behavior, or tick the stages you want help with.",
    sourcePrefix: "from",
    sources: { behavior: "Behavior", issues: "experiment issues", prediction: "prediction", readiness: "MLOps readiness" },
    needSheets: "Add at least one score sheet first.",
    run: "Suggest who can help",
    running: "Working it out…",
    noMatch: "No clear match",
    resultTitle: "Suggestions",
    stages: {
      data_cleaning: "Data cleaning",
      eda: "Exploratory analysis",
      feature_engineering: "Feature engineering",
      model_development: "Model development",
      model_evaluation: "Model evaluation",
      deployment_mlops: "Deployment and MLOps",
      documentation: "Documentation",
    },
  },
} as const

