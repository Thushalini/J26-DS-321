"use client"

import { useCallback, useEffect, useState } from "react"

import { api, ApiError, shortDate } from "@/lib/api"
import { useSession } from "@/lib/session"

type Level = "HIGH" | "MEDIUM" | "LOW"

interface AlertItem {
  run_id: string
  created_at: string
  n_epochs: number | null
  anomaly_score: number | null
  severity_score: number | null
  severity_level: Level
  is_issue: boolean
  resolved: boolean
  title: string | null
  evidence: string | null
  recommended_cause: string | null
  second_cause: string | null
  recommendation: string | null
  root_cause_probabilities: Record<string, number>
}

interface AlertsResponse {
  team_id: string
  alerts: AlertItem[]
}

const CAUSE_LABELS: Record<string, string> = {
  overfitting: "Overfitting",
  stalled: "Stalled experiment",
  data_leakage: "Data leakage",
  resource_bottleneck: "Resource bottleneck",
  premature_evaluation: "Premature evaluation",
}

const LEVEL_STYLE: Record<Level, string> = {
  HIGH: "border-status-critical/40 bg-status-critical/10 text-status-critical",
  MEDIUM: "border-status-warning bg-status-warning/20 text-foreground",
  LOW: "border-status-good/40 bg-status-good/10 text-status-good",
}

function causeLabel(key: string | null): string {
  if (!key) return "None"
  return CAUSE_LABELS[key] ?? key
}

function LevelBadge({ level }: { level: Level }) {
  return (
    <span
      className={`mt-0.5 shrink-0 rounded-bar border px-1.5 py-0.5 font-mono text-[10px] font-semibold ${LEVEL_STYLE[level]}`}
    >
      {level}
    </span>
  )
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 20 20"
      className={`mt-1 h-4 w-4 shrink-0 text-subtle transition-transform ${open ? "rotate-90" : ""}`}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M7 4l6 6-6 6" />
    </svg>
  )
}

function CauseBars({ probs, top }: { probs: Record<string, number>; top: string | null }) {
  const rows = Object.entries(probs).sort((a, b) => b[1] - a[1])
  return (
    <ul className="space-y-1.5">
      {rows.map(([key, p]) => (
        <li key={key} className="flex items-center gap-2 text-xs">
          <span className="w-32 shrink-0 text-muted">{causeLabel(key)}</span>
          <span className="h-2 flex-1 overflow-hidden rounded-bar bg-chart-track">
            <span
              className={`block h-full rounded-bar ${key === top ? "bg-chart-series" : "bg-chart-axis"}`}
              style={{ width: `${Math.round(p * 100)}%` }}
            />
          </span>
          <span className="w-9 text-right tabular-nums text-foreground">{Math.round(p * 100)}%</span>
        </li>
      ))}
    </ul>
  )
}

function StatCard({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="rounded-card border border-border bg-surface p-3">
      <p className="font-mono text-[10px] tracking-wide text-subtle">{label}</p>
      <p className={`mt-1 font-mono text-2xl font-semibold ${tone}`}>{value}</p>
    </div>
  )
}

export default function Page() {
  const { team } = useSession()
  const teamId = team?.team_id ?? null
  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [busyId, setBusyId] = useState<string | null>(null)

  // Alerts belong to the team picked at the top, so load them again whenever that changes.
  useEffect(() => {
    if (!teamId) {
      setAlerts([])
      setLoading(false)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    setExpandedId(null)
    api<AlertsResponse>(`/c3/teams/${encodeURIComponent(teamId)}/alerts`)
      .then((r) => {
        if (!cancelled) setAlerts(r.alerts)
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Could not load the alerts.")
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [teamId])

  const canResolve =
    !!team && !team.anonymised && !team.is_research && (team.my_role === "lead" || team.my_role === "member")

  const toggleResolved = useCallback(
    async (a: AlertItem) => {
      if (!teamId) return
      setBusyId(a.run_id)
      setError(null)
      try {
        const updated = await api<AlertItem>(
          `/c3/teams/${encodeURIComponent(teamId)}/alerts/${encodeURIComponent(a.run_id)}/resolve`,
          { method: "POST", json: { resolved: !a.resolved } },
        )
        setAlerts((prev) => prev.map((x) => (x.run_id === updated.run_id ? updated : x)))
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Could not save the change.")
      } finally {
        setBusyId(null)
      }
    },
    [teamId],
  )

  const issues = alerts.filter((a) => a.is_issue)
  const healthy = alerts.filter((a) => !a.is_issue)
  const openCount = issues.filter((x) => !x.resolved).length
  const resolvedCount = issues.length - openCount

  return (
    <div className="space-y-4">
      <header className="space-y-1">
        <p className="font-mono text-[10px] tracking-wide text-subtle">COMPONENT 3 · ADAPTIVE INTERVENTION</p>
        <h1 className="text-lg font-semibold text-foreground">Anomaly Detection</h1>
        <p className="text-sm text-muted">
          Experiment signals are scored by the Component 3 pipeline. It flags the issue, names the likely root
          cause and suggests an intervention.
        </p>
        {team && <p className="text-xs text-subtle">Alerts for {team.name}</p>}
      </header>

      {error && (
        <p role="alert" className="rounded-card border border-status-critical/40 bg-status-critical/10 p-3 text-sm text-status-critical">
          {error}
        </p>
      )}

      {!teamId && !loading && <p className="text-sm text-muted">Pick a team at the top to see its alerts.</p>}
      {loading && teamId && <p className="text-sm text-muted">Loading alerts…</p>}

      {!loading && teamId && !error && alerts.length === 0 && (
        <div className="rounded-card border border-border bg-surface p-4">
          <p className="text-sm font-semibold text-foreground">No alerts yet for this team</p>
          <p className="mt-1 text-sm text-muted">
            Alerts appear here when the team&rsquo;s experiment runs are scored by Component 3.
          </p>
        </div>
      )}

      {alerts.length > 0 && (
        <>
          <div className="grid grid-cols-2 gap-3">
            <StatCard label="OPEN" value={openCount} tone="text-status-critical" />
            <StatCard label="RESOLVED" value={resolvedCount} tone="text-status-good" />
          </div>

          <ul className="space-y-3">
            {[...issues.filter((a) => !a.resolved), ...issues.filter((a) => a.resolved)].map((a) => {
              const open = expandedId === a.run_id
              return (
                <li
                  key={a.run_id}
                  className={`rounded-card border border-border bg-surface ${a.resolved ? "opacity-60" : ""}`}
                >
                  <button
                    type="button"
                    onClick={() => setExpandedId(open ? null : a.run_id)}
                    aria-expanded={open}
                    className="flex w-full items-start gap-3 p-3 text-left"
                  >
                    <LevelBadge level={a.severity_level} />
                    <span className="min-w-0 flex-1">
                      <span className="block text-sm font-semibold text-foreground">{a.title ?? a.run_id}</span>
                      <span className="mt-0.5 block text-xs text-subtle">
                        {shortDate(a.created_at)}
                        {a.resolved ? " · resolved" : ""}
                      </span>
                    </span>
                    <Chevron open={open} />
                  </button>

                  {open && (
                    <div className="space-y-3 border-t border-border p-3">
                      {a.evidence && <p className="text-sm text-muted">{a.evidence}</p>}
                      <p className="text-xs text-subtle">
                        Anomaly score {(a.anomaly_score ?? 0).toFixed(2)} · Severity{" "}
                        {(a.severity_score ?? 0).toFixed(2)}
                        {a.n_epochs ? ` · ${a.n_epochs} epochs` : ""}
                      </p>

                      <div className="rounded-card border border-border bg-background p-3">
                        <p className="font-mono text-[10px] tracking-wide text-subtle">LIKELY ROOT CAUSE</p>
                        <p className="mt-1 text-sm font-semibold text-status-critical">
                          {causeLabel(a.recommended_cause)}
                        </p>
                        {a.second_cause && (
                          <p className="text-xs text-muted">Also possible: {causeLabel(a.second_cause)}</p>
                        )}
                        <div className="mt-3">
                          <CauseBars probs={a.root_cause_probabilities} top={a.recommended_cause} />
                        </div>
                      </div>

                      {a.recommendation && (
                        <div className="rounded-card border border-primary/30 bg-primary/5 p-3">
                          <p className="font-mono text-[10px] tracking-wide text-subtle">SUGGESTED INTERVENTION</p>
                          <p className="mt-1 text-sm text-foreground">{a.recommendation}</p>
                        </div>
                      )}

                      {canResolve && (
                        <button
                          type="button"
                          disabled={busyId === a.run_id}
                          onClick={() => toggleResolved(a)}
                          className={
                            a.resolved
                              ? "w-full rounded-card border border-border px-3 py-2 text-sm font-semibold text-foreground disabled:opacity-50"
                              : "w-full rounded-card bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground disabled:opacity-50"
                          }
                        >
                          {a.resolved ? "Reopen" : "Mark resolved"}
                        </button>
                      )}
                    </div>
                  )}
                </li>
              )
            })}
          </ul>

          {healthy.length > 0 && (
            <section aria-labelledby="healthy-heading" className="space-y-2">
              <h2 id="healthy-heading" className="font-mono text-[10px] tracking-wide text-subtle">
                HEALTHY RUNS · NO ACTION NEEDED
              </h2>
              <ul className="divide-y divide-border rounded-card border border-border bg-surface">
                {healthy.map((a) => (
                  <li key={a.run_id} className="flex items-center justify-between gap-3 p-3 text-sm">
                    <span className="flex items-center gap-2">
                      <span
                        className={`rounded-bar border px-1.5 py-0.5 font-mono text-[10px] font-semibold ${LEVEL_STYLE.LOW}`}
                      >
                        OK
                      </span>
                      <span className="text-foreground">{a.run_id}</span>
                    </span>
                    <span className="text-xs text-muted">anomaly score {(a.anomaly_score ?? 0).toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  )
}
