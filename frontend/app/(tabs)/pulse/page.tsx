"use client"

import Link from "next/link"

import { NotEnoughData, PredictionState, ProbabilityCard, SignalsNote } from "@/components/predict/prediction-sections"
import { getNavItem } from "@/lib/navigation"
import { predictCopy } from "@/lib/predict/config"
import { useTeamPrediction } from "@/lib/predict/use-team-prediction"
import { useSession } from "@/lib/session"

export default function PulsePage() {
  const { team } = useSession()
  const { prediction, error, loading, reload } = useTeamPrediction(team?.team_id)

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">{predictCopy.pulse.title}</h1>
      <PredictionState loading={loading} error={error} onRetry={reload} />
      {prediction && (
        <>
          {prediction.enoughData ? <ProbabilityCard prediction={prediction} /> : <NotEnoughData prediction={prediction} />}
          <SignalsNote prediction={prediction} />
          <Link href={getNavItem("predict").href} className="block text-xs text-primary">{predictCopy.pulse.more}</Link>
        </>
      )}
    </div>
  )
}
