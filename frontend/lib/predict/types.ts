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
  } | null
  recommendations: Recommendation[]
  enoughData: boolean
  signalsFromTeam?: number
  totalSignals?: number
  minSignals?: number
  laggingStages?: string[]
  weakStages?: WeakStage[]
  changes?: PredictionChanges | null
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

export type HelperSuggestion = {
  stage: string
  stage_label: string
  member_id: string | null
  member_name: string | null
  reason: string
}
