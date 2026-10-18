"use client"

import type React from "react"
import { useState } from "react"
import { formatPercent } from "@/lib/format"
import { forecastChart, predictCopy, scoreMax } from "@/lib/predict/config"
import type { ForecastPoint } from "@/lib/predict/types"

const copy = predictCopy.forecast
const { units } = predictCopy

type BandKey = "p10" | "p25" | "p75" | "p90"

function weekLabel(week: number): string {
  return week === 0 ? units.now : `${units.week} ${week}`
}

export function ForecastCone({ points }: { points: ForecastPoint[] }) {
  const { width, height, margin, yTicks, xTickEvery, dotRadius, labelGap } = forecastChart
  const lastIndex = points.length - 1
  const [activeIndex, setActiveIndex] = useState(lastIndex)

  if (points.length === 0) return null

  const firstWeek = points[0].week
  const span = points[lastIndex].week - firstWeek || 1
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom
  const plotBottom = height - margin.bottom

  const x = (week: number) => margin.left + ((week - firstWeek) / span) * plotWidth
  const y = (value: number) => margin.top + (1 - value / scoreMax) * plotHeight

  const band = (low: BandKey, high: BandKey) => {
    const upper = points.map((point) => `${x(point.week)},${y(point[high])}`)
    const lower = [...points].reverse().map((point) => `${x(point.week)},${y(point[low])}`)
    return `M ${[...upper, ...lower].join(" L ")} Z`
  }
  const median = `M ${points.map((point) => `${x(point.week)},${y(point.p50)}`).join(" L ")}`

  const active = points[activeIndex]
  const end = points[lastIndex]

  const moveTo = (index: number) => setActiveIndex(Math.min(Math.max(index, 0), lastIndex))

  const handlePointer = (event: React.PointerEvent<SVGRectElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    moveTo(Math.round(((event.clientX - bounds.left) / bounds.width) * lastIndex))
  }

  const handleKey = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowLeft") moveTo(activeIndex - 1)
    if (event.key === "ArrowRight") moveTo(activeIndex + 1)
  }

  return (
    <div>
      <p className="text-xs text-muted" aria-live="polite">
        <span className="font-medium text-foreground">{weekLabel(active.week)}</span>
        {" · "}
        {copy.median} {formatPercent(active.p50)}
        {" · "}
        {copy.range} {Math.round(active.p10)}–{formatPercent(active.p90)}
      </p>

      <div tabIndex={0} onKeyDown={handleKey} aria-label={predictCopy.forecast.title} className="mt-2 rounded-bar">
        <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img" aria-label={copy.description}>
          {yTicks.map((tick) => (
            <g key={tick}>
              <line x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} className="stroke-chart-grid" />
              <text
                x={margin.left - labelGap}
                y={y(tick)}
                dy="0.32em"
                textAnchor="end"
                className="fill-subtle text-chart tabular-nums"
              >
                {formatPercent(tick)}
              </text>
            </g>
          ))}

          {points
            .filter((_, index) => index % xTickEvery === 0)
            .map((point) => (
              <text
                key={point.week}
                x={x(point.week)}
                y={height - labelGap}
                textAnchor="middle"
                className="fill-subtle text-chart"
              >
                {weekLabel(point.week)}
              </text>
            ))}

          <path d={band("p10", "p90")} className="fill-chart-series/10" />
          <path d={band("p25", "p75")} className="fill-chart-series/15" />
          <path
            d={median}
            fill="none"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            className="stroke-chart-series"
          />

          <line
            x1={x(active.week)}
            x2={x(active.week)}
            y1={margin.top}
            y2={plotBottom}
            className="stroke-chart-axis"
          />
          <circle
            cx={x(active.week)}
            cy={y(active.p50)}
            r={dotRadius}
            strokeWidth={2}
            className="fill-chart-series stroke-surface"
          />

          <text
            x={x(end.week) + labelGap}
            y={y(end.p50)}
            dy="0.32em"
            className="fill-foreground text-chart font-medium tabular-nums"
          >
            {formatPercent(end.p50)}
          </text>

          <rect
            x={margin.left}
            y={margin.top}
            width={plotWidth}
            height={plotHeight}
            fill="transparent"
            className="touch-pan-y"
            onPointerMove={handlePointer}
            onPointerDown={handlePointer}
          />
        </svg>
      </div>

      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        <li className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded-full bg-chart-series" />
          {copy.median}
        </li>
        <li className="flex items-center gap-1.5">
          <span className="size-2.5 rounded-bar bg-chart-series/25" />
          {copy.inner}
        </li>
        <li className="flex items-center gap-1.5">
          <span className="size-2.5 rounded-bar bg-chart-series/10" />
          {copy.outer}
        </li>
      </ul>
    </div>
  )
}
