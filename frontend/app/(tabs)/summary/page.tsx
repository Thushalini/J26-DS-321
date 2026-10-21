"use client"

import {
  ForecastCard, NotEnoughData, PredictionState, ProbabilityCard, ReadinessCard, SignalsNote,
} from "@/components/predict/prediction-sections"
import { predictCopy } from "@/lib/predict/config"
import { useTeamPrediction } from "@/lib/predict/use-team-prediction"
import { useSession } from "@/lib/session"

export default function SummaryPage() {
  const { team } = useSession()
  const { prediction, error, loading, reload } = useTeamPrediction(team?.team_id)

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-lg font-semibold">{predictCopy.summary.title}</h1>
        <p className="text-xs text-muted">{predictCopy.summary.description}</p>
      </header>
      {!team && <p className="text-sm text-muted">{predictCopy.live.noTeam}</p>}
      <PredictionState loading={loading} error={error} onRetry={reload} />
      {prediction && !prediction.enoughData && <NotEnoughData prediction={prediction} />}
      {prediction?.enoughData && (
        <>
          <SignalsNote prediction={prediction} />
          <ProbabilityCard prediction={prediction} />
          <ForecastCard prediction={prediction} />
        </>
      )}
      {prediction && <ReadinessCard prediction={prediction} />}
    </div>
  )
}
