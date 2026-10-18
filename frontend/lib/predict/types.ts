export type StatusLevel = "good" | "warning" | "critical"

export type Factor = {
  id: string
  label: string
  contribution: number
  detail: string
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
  }
  recommendations: Recommendation[]
}
