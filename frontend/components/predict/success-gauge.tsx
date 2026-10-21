import { IconArrowDown, IconArrowUp } from "@/components/icons"
import { StatusBadge } from "@/components/predict/status-badge"
import { formatPercent, formatSigned } from "@/lib/format"
import { gauge, pointsDecimals, predictCopy, scoreMax, sparkline, statusLabels } from "@/lib/predict/config"
import { getStatusLevel } from "@/lib/predict/status"

const copy = predictCopy.probability

function Gauge({ value }: { value: number }) {
  const { radius, strokeWidth } = gauge
  const width = radius * 2 + strokeWidth
  const height = radius + strokeWidth
  const baseline = radius + strokeWidth / 2
  const arc = `M ${strokeWidth / 2} ${baseline} A ${radius} ${radius} 0 0 1 ${width - strokeWidth / 2} ${baseline}`

  return (
    <div className="relative mx-auto" style={{ maxWidth: width }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-auto w-full"
        role="img"
        aria-label={`${copy.title}: ${formatPercent(value)}`}
      >
        <path d={arc} fill="none" strokeWidth={strokeWidth} strokeLinecap="round" className="stroke-chart-track" />
        <path
          d={arc}
          fill="none"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          pathLength={scoreMax}
          strokeDasharray={`${value} ${scoreMax}`}
          className="stroke-chart-series"
        />
      </svg>
      <p className="absolute inset-x-0 bottom-0 text-center text-5xl font-semibold">{formatPercent(value)}</p>
    </div>
  )
}

function Sparkline({ values }: { values: number[] }) {
  const { width, height, padding, dotRadius } = sparkline
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const step = (width - padding * 2) / Math.max(values.length - 1, 1)
  const points = values.map((value, index) => ({
    x: padding + index * step,
    y: padding + (1 - (value - min) / range) * (height - padding * 2),
  }))
  const last = points[points.length - 1]

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} aria-hidden="true">
      <polyline
        points={points.map((point) => `${point.x},${point.y}`).join(" ")}
        fill="none"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="stroke-chart-axis"
      />
      <circle cx={last.x} cy={last.y} r={dotRadius} strokeWidth={2} className="fill-chart-series stroke-surface" />
    </svg>
  )
}

export function SuccessGauge({ probability, history }: { probability: number; history: number[] }) {
  const previous = history[history.length - 2]
  const delta = previous === undefined ? 0 : Number((probability - previous).toFixed(pointsDecimals))
  const DeltaIcon = delta < 0 ? IconArrowDown : IconArrowUp
  const level = getStatusLevel(probability)

  return (
    <div>
      <Gauge value={probability} />
      <div className="mt-4 flex items-end justify-between gap-3">
        <div className="space-y-2">
          <StatusBadge level={level} label={statusLabels.probability[level]} />
          {delta !== 0 && (
            <p className="flex items-center gap-1 text-xs text-muted">
              <DeltaIcon className={`size-3.5 ${delta < 0 ? "text-status-critical" : "text-status-good"}`} />
              <span className="font-medium text-foreground">
                {formatSigned(delta)} {predictCopy.units.points}
              </span>
              {copy.delta}
            </p>
          )}
        </div>
        {history.length > 1 && (
          <figure className="shrink-0 text-right">
            <Sparkline values={history} />
            <figcaption className="text-chart text-subtle">{copy.history}</figcaption>
          </figure>
        )}
      </div>
    </div>
  )
}
