"use client"

import { HelperSuggestions } from "@/components/predict/helper-suggestions"
import {
  FactorsCard, ForecastCard, NotEnoughData, PredictionState, ProbabilityCard, ReadinessCard, SignalsNote,
} from "@/components/predict/prediction-sections"
import { Recommendations } from "@/components/predict/recommendations"
import { SectionCard } from "@/components/section-card"
import { formatDate } from "@/lib/format"
import { predictCopy } from "@/lib/predict/config"
import { useTeamPrediction } from "@/lib/predict/use-team-prediction"
import { useSession } from "@/lib/session"

export default function PredictPage() {
  const { team } = useSession()
  const { prediction, error, loading, reload } = useTeamPrediction(team?.team_id)
  const canUploadSheets = !team?.is_research && ["lead", "member", "supervisor"].includes(team?.my_role ?? "")

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-lg font-semibold">{predictCopy.title}</h1>
        {prediction?.enoughData && (
          <p className="text-xs text-muted">
            {predictCopy.evaluated} {formatDate(prediction.evaluatedAt)}
          </p>
        )}
      </header>

      <PredictionState loading={loading} error={error} onRetry={reload} />

      {prediction && !prediction.enoughData && <NotEnoughData prediction={prediction} />}

      {prediction?.enoughData && (
        <>
          <SignalsNote prediction={prediction} />
          <ProbabilityCard prediction={prediction} />
          <FactorsCard prediction={prediction} />
          <ForecastCard prediction={prediction} />
        </>
      )}

      {prediction && <ReadinessCard prediction={prediction} />}

      {prediction?.enoughData && (
        <SectionCard title={predictCopy.recommendations.title} description={predictCopy.recommendations.description}>
          <Recommendations items={prediction.recommendations} />
        </SectionCard>
      )}

      {team && canUploadSheets && (
        <HelperSuggestions teamId={team.team_id} weakStages={prediction?.weakStages ?? []} />
      )}
    </div>
  )
}
