import { FactorChart } from "@/components/predict/factor-chart"
import { ForecastCone } from "@/components/predict/forecast-cone"
import { Readiness, ReadinessScore } from "@/components/predict/readiness"
import { Recommendations } from "@/components/predict/recommendations"
import { SuccessGauge } from "@/components/predict/success-gauge"
import { SectionCard } from "@/components/section-card"
import { formatDate } from "@/lib/format"
import { predictCopy } from "@/lib/predict/config"
import { getPrediction } from "@/lib/predict/get-prediction"

export default async function Page() {
  const prediction = await getPrediction()

  return (
    <div className="space-y-4">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold">{predictCopy.title}</h1>
          <p className="text-xs text-muted">
            {predictCopy.evaluated} {formatDate(prediction.evaluatedAt)}
          </p>
        </div>
        {prediction.source === "sample" && (
          <span className="shrink-0 rounded-full border border-border bg-surface px-2 py-0.5 text-xs text-muted">
            {predictCopy.sampleBadge}
          </span>
        )}
      </header>

      <SectionCard title={predictCopy.probability.title} description={predictCopy.probability.description}>
        <SuccessGauge probability={prediction.probability} history={prediction.history} />
      </SectionCard>

      <SectionCard title={predictCopy.factors.title} description={predictCopy.factors.description}>
        <FactorChart factors={prediction.factors} />
      </SectionCard>

      <SectionCard title={predictCopy.forecast.title} description={predictCopy.forecast.description}>
        <ForecastCone points={prediction.forecast} />
      </SectionCard>

      <SectionCard
        title={predictCopy.readiness.title}
        description={predictCopy.readiness.description}
        aside={<ReadinessScore score={prediction.readiness.score} />}
      >
        <Readiness score={prediction.readiness.score} areas={prediction.readiness.areas} />
      </SectionCard>

      <SectionCard title={predictCopy.recommendations.title} description={predictCopy.recommendations.description}>
        <Recommendations items={prediction.recommendations} />
      </SectionCard>
    </div>
  )
}
