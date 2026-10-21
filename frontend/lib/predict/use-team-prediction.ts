"use client"

import { useCallback, useEffect, useState } from "react"

import { api } from "@/lib/api"
import type { ForecastPoint, Prediction, ReadinessArea, WeakStage } from "@/lib/predict/types"

type ApiPrediction = {
  source: "live" | "synthetic"
  enough_data?: boolean
  evaluated_at: string
  probability: number | null
  factors: { id: string; label: string; contribution: number; detail: string; measured?: boolean }[]
  forecast: ForecastPoint[]
  recommendations: { id: string; title: string; reason: string; area: string; expected_gain: number }[]
  readiness: { score: number; areas: ReadinessArea[] } | null
  imputed_features: string[]
  history?: number[]
  changes?: { since: string; probability_change: number; signals: { id: string; label: string; before: number; after: number }[] } | null
  signals_from_team?: number
  total_signals?: number
  min_signals?: number
  lagging_stages?: string[]
  weak_stages?: WeakStage[]
}

export function toPrediction(body: ApiPrediction): Prediction {
  return {
    source: "live",
    evaluatedAt: body.evaluated_at,
    probability: body.probability ?? 0,
    enoughData: body.enough_data ?? true,
    history: body.history ?? [],
    factors: body.factors.map(({ id, label, contribution, detail, measured }) => ({ id, label, contribution, detail, measured })),
    forecast: body.forecast,
    readiness: body.readiness && {
      score: body.readiness.score,
      areas: body.readiness.areas.map(({ id, label, score, reason }) => ({ id, label, score, reason })),
    },
    changes: body.changes && {
      since: body.changes.since, probabilityChange: body.changes.probability_change, signals: body.changes.signals,
    },
    recommendations: body.recommendations.map((r) => ({
      id: r.id, title: r.title, reason: r.reason, area: r.area, expectedGain: r.expected_gain,
    })),
    signalsFromTeam: body.signals_from_team,
    totalSignals: body.total_signals ?? body.factors.length,
    minSignals: body.min_signals,
    laggingStages: body.lagging_stages ?? [],
    weakStages: body.weak_stages ?? [],
  }
}

export function useTeamPrediction(teamId: string | undefined) {
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    if (!teamId) return
    setLoading(true)
    setError("")
    try {
      setPrediction(toPrediction(await api<ApiPrediction>(`/c4/teams/${encodeURIComponent(teamId)}/prediction`)))
    } catch (e) {
      setPrediction(null)
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [teamId])

  useEffect(() => {
    load()
  }, [load])

  return { prediction, error, loading, reload: load }
}
