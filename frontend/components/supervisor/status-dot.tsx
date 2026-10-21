import { statusInfo, type SupervisionStatus } from "@/lib/supervision"

export function StatusDot({ status, manual = false, title }: {
  status: SupervisionStatus | null | undefined
  manual?: boolean
  title?: string
}) {
  const info = statusInfo(status)
  const label = title ?? info.label
  return (
    <span className="inline-flex shrink-0 items-center" title={label}>
      <span className={`size-2.5 rounded-full ${info.dot} ${manual ? "ring-2 ring-primary ring-offset-1 ring-offset-surface" : ""}`}
            aria-hidden="true" />
      <span className="sr-only">{label}</span>
    </span>
  )
}
