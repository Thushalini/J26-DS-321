"use client"

import { useState } from "react"
import { formatSigned } from "@/lib/format"
import { predictCopy } from "@/lib/predict/config"
import type { Factor } from "@/lib/predict/types"

const copy = predictCopy.factors

export function FactorChart({ factors }: { factors: Factor[] }) {
  const sorted = [...factors].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
  const largest = Math.max(...sorted.map((factor) => Math.abs(factor.contribution)), 1)
  const [selectedId, setSelectedId] = useState(sorted[0]?.id)
  const selected = sorted.find((factor) => factor.id === selectedId)

  return (
    <div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        <li className="flex items-center gap-1.5">
          <span className="size-2.5 rounded-bar bg-chart-positive" />
          {copy.raises}
        </li>
        <li className="flex items-center gap-1.5">
          <span className="size-2.5 rounded-bar bg-chart-negative" />
          {copy.lowers}
        </li>
      </ul>

      <ul className="mt-3 space-y-1">
        {sorted.map((factor) => {
          const positive = factor.contribution >= 0
          const share = (Math.abs(factor.contribution) / largest) * 50
          const active = factor.id === selectedId

          return (
            <li key={factor.id}>
              <button
                type="button"
                onClick={() => setSelectedId(factor.id)}
                aria-pressed={active}
                className={`w-full rounded-bar px-2 py-1.5 text-left hover:bg-background ${active ? "bg-background" : ""}`}
              >
                <span className="flex items-baseline justify-between gap-2 text-xs">
                  <span>{factor.label}</span>
                  <span className="shrink-0 tabular-nums text-muted">
                    {formatSigned(factor.contribution)} {predictCopy.units.points}
                  </span>
                </span>
                <span className="relative mt-1.5 block h-3">
                  <span className="absolute inset-y-0 left-1/2 w-px bg-chart-axis" />
                  <span
                    className={`absolute inset-y-0 ${
                      positive ? "left-1/2 rounded-r-bar bg-chart-positive" : "right-1/2 rounded-l-bar bg-chart-negative"
                    }`}
                    style={{ width: `${share}%` }}
                  />
                </span>
              </button>
            </li>
          )
        })}
      </ul>

      {selected && (
        <p className="mt-3 border-t border-border pt-3 text-xs text-muted" aria-live="polite">
          <span className="font-medium text-foreground">{selected.label}.</span> {selected.detail}
        </p>
      )}
    </div>
  )
}
