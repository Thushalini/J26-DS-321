import { formatSigned } from "@/lib/format"
import { predictCopy } from "@/lib/predict/config"
import type { Recommendation } from "@/lib/predict/types"

const copy = predictCopy.recommendations

export function Recommendations({ items }: { items: Recommendation[] }) {
  return (
    <ol className="space-y-4">
      {items.map((item, index) => (
        <li key={item.id} className="flex gap-3">
          <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-background text-xs font-semibold">
            {index + 1}
          </span>
          <div className="min-w-0 space-y-1.5">
            <p className="text-sm font-medium">{item.title}</p>
            <p className="text-xs text-muted">{item.reason}</p>
            <p className="flex flex-wrap items-center gap-2 text-xs">
              <span className="rounded-full border border-border px-2 py-0.5 text-muted">{item.area}</span>
              <span className="font-medium tabular-nums">
                {formatSigned(item.expectedGain)} {predictCopy.units.points}
              </span>
              <span className="text-muted">{copy.gain}</span>
            </p>
            {item.support && (
              <p className="text-xs text-muted">
                {copy.support}: <span className="font-medium text-foreground">{item.support.member}</span>,{" "}
                {copy.strength} {item.support.strength}
              </p>
            )}
          </div>
        </li>
      ))}
    </ol>
  )
}
