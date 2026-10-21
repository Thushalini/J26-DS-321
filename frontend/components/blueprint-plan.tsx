"use client"
// The nine-field planning blueprint built by Component 1 (Buddy) from the team chat.
import { useEffect, useState } from "react"

import { Card, Notice } from "@/components/ui"
import { getPlan, shortDate, type PlanData, type PlanField } from "@/lib/api"

const POLL_MS = 10000

const STATUS: Record<string, { text: string; cls: string }> = {
  missing: { text: "Not discussed yet", cls: "border-border text-muted" },
  uncertain: { text: "Not sure yet", cls: "border-status-warning text-foreground bg-status-warning/10" },
  known: { text: "Mentioned", cls: "border-primary text-primary bg-primary/10" },
  confirmed: { text: "Confirmed", cls: "border-status-good text-status-good bg-status-good/10" },
  conflicting: { text: "Team disagrees", cls: "border-status-critical text-status-critical bg-status-critical/10" },
}

function StatusBadge({ status }: { status: string }) {
  const s = STATUS[status] ?? { text: status, cls: "border-border text-muted" }
  return <span className={`rounded-full border px-2 py-0.5 text-xs ${s.cls}`}>{s.text}</span>
}

function FieldCard({ f, showSources }: { f: PlanField; showSources: boolean }) {
  return (
    <li className="rounded-card border border-border bg-surface px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">{f.label}</h3>
        <StatusBadge status={f.status} />
      </div>

      {f.status === "conflicting" && f.candidates?.length ? (
        <div className="mt-1">
          <p className="text-xs text-muted">Different answers so far:</p>
          <ul className="mt-1 flex flex-col gap-1">
            {f.candidates.map((c) => (
              <li key={c} className="rounded-bar border border-border px-2 py-1 text-sm">{c}</li>
            ))}
          </ul>
        </div>
      ) : f.value ? (
        <p className="mt-1 text-sm">{f.value}</p>
      ) : (
        <p className="mt-1 text-sm text-muted">Nothing recorded yet</p>
      )}

      {f.confidence_band && (
        <p className="mt-1 text-xs text-muted">
          Confidence: {f.confidence_band} · mentioned {f.mentions} {f.mentions === 1 ? "time" : "times"}
        </p>
      )}

      {showSources && f.evidence && f.evidence.length > 0 && (
        <details className="mt-1">
          <summary className="cursor-pointer text-xs text-primary">What was said</summary>
          <ul className="mt-1 flex flex-col gap-1">
            {f.evidence.map((e, i) => (
              <li key={`${e.source_id}-${i}`} className="border-l-2 border-border pl-2 text-xs text-muted">
                <span className="font-semibold text-foreground">{e.speaker}</span>
                {e.time ? ` · ${shortDate(e.time)}` : ""}: &ldquo;{e.text}&rdquo;
              </li>
            ))}
          </ul>
        </details>
      )}
    </li>
  )
}

export function BlueprintPlan({ teamId }: { teamId: string }) {
  const [plan, setPlan] = useState<PlanData | null>(null)
  const [unavailable, setUnavailable] = useState("")
  const [error, setError] = useState("")

  useEffect(() => {
    let alive = true
    setPlan(null)
    setUnavailable("")
    setError("")
    const load = () =>
      getPlan(teamId)
        .then((p) => {
          if (!alive) return
          if (p.available) { setPlan(p); setUnavailable("") }
          else setUnavailable(p.reason)
          setError("")
        })
        .catch((e) => alive && setError(e.message))
    load()
    const id = setInterval(load, POLL_MS)
    return () => { alive = false; clearInterval(id) }
  }, [teamId])

  if (unavailable) return null   // research teams: only the old blueprint card is shown
  if (error) return <Notice tone="error">{error}</Notice>
  if (!plan) return <p className="text-sm text-muted">Loading the planning blueprint...</p>

  const r = plan.readiness
  const pct = r.total ? Math.round((r.confirmed / r.total) * 100) : 0

  return (
    <div className="flex flex-col gap-3">
      <Card
        title="Planning blueprint (built by Buddy from your chat)"
        note="Buddy listens to the team chat and fills these nine fields. Nothing here is final until the team confirms it."
      >
        <div className="flex flex-col gap-2">
          <p className="text-sm">
            <span className="font-semibold">{r.confirmed} of {r.total}</span> fields confirmed
            {r.complete ? " · ready to build" : ""}
          </p>
          <div className="h-2 w-full overflow-hidden rounded-bar bg-chart-track" role="progressbar"
               aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full bg-primary" style={{ width: `${pct}%` }} />
          </div>
          {r.conflicting.length > 0 && (
            <p className="text-xs text-status-critical">
              The team gave different answers for: {r.conflicting.join(", ").replaceAll("_", " ")}.
            </p>
          )}
          {plan.next_gaps.length > 0 && (
            <p className="text-xs text-muted">
              Buddy will ask about next: {plan.next_gaps.join(", ").replaceAll("_", " ")}.
            </p>
          )}
          {plan.updated_at && <p className="text-xs text-muted">Last change {shortDate(plan.updated_at)}</p>}
        </div>
      </Card>

      <ul className="flex flex-col gap-2">
        {plan.fields.map((f) => (
          <FieldCard key={f.name} f={f} showSources={plan.can_see_sources} />
        ))}
      </ul>

      {!plan.can_see_sources && (
        <p className="text-xs text-muted">
          You can see values and status only. The quoted chat words are private to the team lead and members.
        </p>
      )}
      <p className="text-xs text-muted">{plan.note}</p>
    </div>
  )
}