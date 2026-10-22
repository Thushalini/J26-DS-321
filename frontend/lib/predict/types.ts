export type StatusLevel = "good" | "warning" | "critical"

export type Factor = {
  id: string
  label: string
  contribution: number
  detail: string
  measured?: boolean
}

export type ForecastPoint = {
  week: number
  p10: number
  p25: number
  p50: number
  p75: number
  p90: number
}

export type ReadinessArea = {
  id: string
  label: string
  score: number
  reason: string
}

export type Recommendation = {
  id: string
  title: string
  reason: string
  area: string
  expectedGain: number
  support?: {
    member: string
    strength: string
  }
}

export type Prediction = {
  source: "sample" | "live"
  evaluatedAt: string
  probability: number
  history: number[]
  factors: Factor[]
  forecast: ForecastPoint[]
  readiness: {
    score: number
    areas: ReadinessArea[]
    basis?: "contents" | "names"
  } | null
  recommendations: Recommendation[]
  enoughData: boolean
  probabilityRange?: { low: number; high: number }
  signalsFromTeam?: number
  totalSignals?: number
  minSignals?: number
  laggingStages?: string[]
  weakStages?: WeakStage[]
  changes?: PredictionChanges | null
  milestones?: Milestone[]
  timelineBasis?: TimelineBasis | null
}

export type Milestone = {
  name: string
  date: string
  stages: string[]
  work?: string | null
  source?: "schedule" | "group"
  due?: boolean
  met?: boolean | null
}

export type TimelineBasis = {
  start: string | null
  startFrom: "repo" | "first_commit" | "group" | null
  schedule: string | null
  scheduleStatus: string | null
}

export type PredictionChanges = {
  since: string
  probabilityChange: number
  signals: { id: string; label: string; before: number; after: number }[]
}

export type WeakStage = {
  stage: string
  label: string
  sources: ("behavior" | "issues" | "prediction" | "readiness")[]
}

export type SavedSheet = {
  user_id: string
  name: string
  saved: boolean
  uploaded_at: string | null
}

export type MilestoneResult = {
  result_id: string
  milestone: string
  date: string
  result: string
  recorded_at: string
}

export type HelperSuggestion = {
  stage: string
  stage_label: string
  member_id: string | null
  member_name: string | null
  reason: string
}
