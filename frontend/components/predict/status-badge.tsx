import { IconAlert, IconCheck, IconCross } from "@/components/icons"
import type { StatusLevel } from "@/lib/predict/types"

const icons = {
  good: IconCheck,
  warning: IconAlert,
  critical: IconCross,
}

const tones: Record<StatusLevel, string> = {
  good: "text-status-good",
  warning: "text-status-warning",
  critical: "text-status-critical",
}

export function StatusBadge({ level, label }: { level: StatusLevel; label: string }) {
  const Icon = icons[level]

  return (
    <span className="inline-flex shrink-0 items-center gap-1 rounded-full border border-border px-2 py-0.5 text-xs font-medium">
      <Icon className={`size-3.5 ${tones[level]}`} />
      {label}
    </span>
  )
}
