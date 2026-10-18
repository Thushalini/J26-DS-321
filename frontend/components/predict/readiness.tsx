import { StatusBadge } from "@/components/predict/status-badge"
import { predictCopy, scoreMax, statusLabels } from "@/lib/predict/config"
import { getStatusLevel } from "@/lib/predict/status"
import type { ReadinessArea } from "@/lib/predict/types"

const copy = predictCopy.readiness

export function ReadinessScore({ score }: { score: number }) {
  const level = getStatusLevel(score)
  return <StatusBadge level={level} label={statusLabels.readiness[level]} />
}

export function Readiness({ score, areas }: { score: number; areas: ReadinessArea[] }) {
  return (
    <div>
      <p>
        <span className="text-3xl font-semibold">{score}</span>{" "}
        <span className="text-xs text-muted">
          {copy.outOf} {scoreMax}
        </span>
      </p>

      <ul className="mt-4 space-y-4">
        {areas.map((area) => {
          const level = getStatusLevel(area.score)

          return (
            <li key={area.id}>
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-medium">{area.label}</span>
                <span className="flex shrink-0 items-center gap-2">
                  <StatusBadge level={level} label={statusLabels.readiness[level]} />
                  <span className="w-6 text-right text-xs tabular-nums">{area.score}</span>
                </span>
              </div>
              <div className="mt-1.5 h-2 rounded-full bg-chart-track">
                <div
                  className="h-full rounded-full bg-chart-series"
                  style={{ width: `${(area.score / scoreMax) * 100}%` }}
                />
              </div>
              <p className="mt-1.5 text-xs text-muted">{area.reason}</p>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
