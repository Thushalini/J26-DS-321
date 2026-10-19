import { statusInfo, type SupervisionStatus } from "@/lib/supervision"

export function StatusDot({ status }: { status: SupervisionStatus | null | undefined }) {
  const info = statusInfo(status)
  return (
    <span className="inline-flex shrink-0 items-center" title={info.label}>
      <span className={`size-2.5 rounded-full ${info.dot}`} aria-hidden="true" />
      <span className="sr-only">{info.label}</span>
    </span>
  )
}
