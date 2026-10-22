import { FactorChart } from "@/components/predict/factor-chart"
import { ForecastCone } from "@/components/predict/forecast-cone"
import { Readiness, ReadinessScore } from "@/components/predict/readiness"
import { SuccessGauge } from "@/components/predict/success-gauge"
import { SectionCard } from "@/components/section-card"
import { Button, Notice } from "@/components/ui"
import { formatDate } from "@/lib/format"
import { predictCopy } from "@/lib/predict/config"
import type { Prediction } from "@/lib/predict/types"

const live = predictCopy.live

function fill(text: string, values: Record<string, string | number | undefined>) {
  return Object.entries(values).reduce((out, [key, value]) => out.replace(`{${key}}`, String(value ?? "")), text)
}

const rounded = (value: number) => Number(value.toFixed(live.decimals))

export function SignalsNote({ prediction }: { prediction: Prediction }) {
  if (!prediction.enoughData || prediction.signalsFromTeam == null || prediction.signalsFromTeam === prediction.totalSignals) return null
  const text = fill(live.signals, { n: prediction.signalsFromTeam, total: prediction.totalSignals })
  return <p className="rounded-card border border-border bg-surface px-3 py-2 text-xs text-muted">{text}</p>
}

export function NotEnoughData({ prediction }: { prediction: Prediction }) {
  return (
    <SectionCard title={live.notEnoughTitle}>
      <p className="text-sm text-muted">
        {fill(live.notEnough, { min: prediction.minSignals, total: prediction.totalSignals, n: prediction.signalsFromTeam })}
      </p>
    </SectionCard>
  )
}

export function PredictionState({ loading, error, onRetry }: { loading: boolean; error: string; onRetry: () => void }) {
  if (error) {
    return (
      <div className="space-y-2">
        <Notice tone="error">{error}</Notice>
        <Button variant="secondary" onClick={onRetry}>{live.retry}</Button>
      </div>
    )
  }
  return loading ? <p className="text-sm text-muted">{live.loading}</p> : null
}

function RangeNote({ prediction }: { prediction: Prediction }) {
  const range = prediction.probabilityRange
  if (!range || range.high - range.low < predictCopy.probability.rangeMinWidth) return null
  return (
    <p className="mt-3 text-xs text-muted">
      {fill(predictCopy.probability.range, { low: Math.round(range.low), high: Math.round(range.high) })}
    </p>
  )
}

function ChangesNote({ prediction }: { prediction: Prediction }) {
  const signals = prediction.changes?.signals ?? []
  if (!prediction.changes || signals.length === 0) return null
  return (
    <p className="mt-3 text-xs text-muted">
      {fill(live.changed, { date: formatDate(prediction.changes.since) })}{" "}
      {signals.map((s) => `${s.label} ${rounded(s.before)} → ${rounded(s.after)}`).join(" · ")}
    </p>
  )
}

export function ProbabilityCard({ prediction }: { prediction: Prediction }) {
  return (
    <SectionCard title={predictCopy.probability.title} description={predictCopy.probability.description}>
      <SuccessGauge probability={prediction.probability} history={prediction.history} />
      <RangeNote prediction={prediction} />
      <ChangesNote prediction={prediction} />
    </SectionCard>
  )
}

export function FactorsCard({ prediction }: { prediction: Prediction }) {
  const unmeasured = prediction.factors.filter((f) => f.measured === false)
  return (
    <SectionCard title={predictCopy.factors.title} description={predictCopy.factors.description}>
      <FactorChart factors={prediction.factors.filter((f) => f.measured !== false)} />
      {unmeasured.length > 0 && (
        <p className="mt-3 text-xs text-muted">{fill(live.notMeasured, { items: unmeasured.map((f) => f.label).join(", ") })}</p>
      )}
    </SectionCard>
  )
}

export function TimelineCard({ prediction }: { prediction: Prediction }) {
  const items = prediction.milestones ?? []
  if (items.length === 0) return null
  const copy = predictCopy.timeline
  const basis = prediction.timelineBasis
  const basisText = basis?.start && basis.startFrom
    ? fill(basis.schedule ? copy.basis : copy.basisNoSchedule, {
        schedule: basis.schedule ?? "",
        status: copy.scheduleStatus[basis.scheduleStatus ?? ""] ?? basis.scheduleStatus ?? "",
        start: formatDate(basis.start),
        from: copy.startFrom[basis.startFrom],
      })
    : ""
  return (
    <SectionCard title={copy.title} description={copy.description}>
      <ul className="flex flex-col divide-y divide-border">
        {items.map((m) => {
          const unchecked = m.stages.length === 0 && m.source === "schedule" ? copy.planning : copy.noStages
          const state = !m.due ? copy.upcoming : m.met == null ? unchecked : m.met ? copy.met : copy.missed
          const tone = !m.due || m.met == null ? "text-muted" : m.met ? "text-success" : "text-danger"
          return (
            <li key={`${m.name}-${m.date}`} className="flex items-center justify-between gap-2 py-2 text-sm">
              <span className="min-w-0">
                <span className="block truncate font-medium">{m.name}</span>
                {m.work && <span className="block text-xs text-muted">{m.work}</span>}
                <span className="block text-xs text-muted">
                  {formatDate(m.date)}{m.source === "schedule" && ` · ${copy.scheduleTag}`}
                </span>
              </span>
              <span className={`shrink-0 text-xs font-semibold ${tone}`}>{state}</span>
            </li>
          )
        })}
      </ul>
      {basisText && <p className="mt-3 text-xs text-muted">{basisText}</p>}
    </SectionCard>
  )
}

export function ForecastCard({ prediction }: { prediction: Prediction }) {
  return (
    <SectionCard title={predictCopy.forecast.title} description={predictCopy.forecast.description}>
      <ForecastCone points={prediction.forecast} />
    </SectionCard>
  )
}

export function ReadinessCard({ prediction }: { prediction: Prediction }) {
  const readiness = prediction.readiness
  if (!readiness) {
    return (
      <SectionCard title={predictCopy.readiness.title}>
        <p className="text-sm text-muted">{live.readinessNone}</p>
      </SectionCard>
    )
  }
  return (
    <SectionCard
      title={predictCopy.readiness.title}
      description={readiness.basis === "names" ? predictCopy.readiness.namesOnly : predictCopy.readiness.description}
      aside={<ReadinessScore score={readiness.score} />}
    >
      <Readiness score={readiness.score} areas={readiness.areas} />
    </SectionCard>
  )
}
