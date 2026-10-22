"use client"
// The ten-field planning blueprint built by Component 1 (Buddy) from the team chat and from
// uploaded project documents. Values found in documents are suggestions until the team accepts them.
import { useCallback, useEffect, useRef, useState } from "react"

import { Button, Card, Notice } from "@/components/ui"
import {
  decideSuggestion,
  getPlan,
  getPlanHistory,
  removePlanDocument,
  shortDate,
  uploadPlanDocument,
  type PlanData,
  type PlanDocument,
  type PlanField,
  type PlanCheck,
  type PlanDecision,
  type PlanFeasibilityDependency,
  type PlanFeasibilityModel,
  type PlanHistoryEntry,
  type PlanMilestone,
  type PlanSuggestion,
  type PlanTimeline,
} from "@/lib/api"

function dayText(iso: string) {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })
}

function daysLeftText(n: number) {
  if (n === 0) return "today"
  if (n < 0) return `${-n} ${n === -1 ? "day" : "days"} ago`
  return `${n} ${n === 1 ? "day" : "days"} left`
}

const CHECK_LOOK: Record<string, { word: string; mark: string; cls: string }> = {
  pass: { word: "Fits", mark: "✓", cls: "border-status-good text-status-good bg-status-good/10" },
  warning: { word: "Check this", mark: "!", cls: "border-status-warning text-foreground bg-status-warning/10" },
  fail: { word: "Does not fit", mark: "✕", cls: "border-status-critical text-status-critical bg-status-critical/10" },
}

// Decisions are drawn as a list with a status mark (not as boxed cards), so they never look like fields.
const DECISION_LOOK: Record<string, { mark: string; cls: string }> = {
  proposed: { mark: "?", cls: "border-border text-muted bg-background" },
  confirmed: { mark: "✓", cls: "border-primary text-primary bg-primary/10" },
  finalized: { mark: "✓✓", cls: "border-status-good text-status-good bg-status-good/10" },
  pending_fix: { mark: "!", cls: "border-status-warning text-foreground bg-status-warning/20" },
  rejected: { mark: "✕", cls: "border-status-critical text-status-critical bg-status-critical/10" },
  superseded: { mark: "↺", cls: "border-border text-subtle bg-background" },
}
const DECISION_GROUPS: { status: string; title: string; note: string }[] = [
  { status: "pending_fix", title: "Needs re-check", note: "An earlier choice changed, so these may no longer fit." },
  { status: "finalized", title: "Finalized", note: "Locked in by the team." },
  { status: "confirmed", title: "Agreed", note: "Someone other than the proposer agreed." },
  { status: "proposed", title: "Waiting for agreement", note: "Proposed in the chat, nobody has agreed yet." },
]
const TIMELINE_WORD: Record<string, string> = {
  proposed: "proposed", confirmed: "confirmed", finalized: "finalized", rejected: "rejected",
  superseded: "replaced", pending_fix: "flagged for re-check", updated: "updated",
}

function DecisionsCard({ d }: { d: NonNullable<PlanData["decisions"]> }) {
  const others = d.all.filter((x) => !d.current.some((c) => c.decision_id === x.decision_id))
  return (
    <Card
      title="Decisions"
      note="What the team chose in the chat: who proposed it, who agreed, and why. If an earlier choice changes, decisions that depend on it are flagged for a re-check."
    >
      {d.current.length === 0 ? (
        <p className="text-sm text-muted">No decisions yet. They appear as the team proposes and agrees on choices in the chat.</p>
      ) : (
        <div className="flex flex-col gap-4">
          {DECISION_GROUPS.map((g) => {
            const items = d.current.filter((x) => x.status === g.status)
            if (items.length === 0) return null
            return (
              <div key={g.status}>
                <p className="text-xs font-semibold uppercase tracking-wide text-muted">
                  {g.title} <span className="font-normal normal-case">({items.length})</span>
                </p>
                <p className="text-xs text-subtle">{g.note}</p>
                <ul className="mt-1 divide-y divide-border">
                  {items.map((x) => <DecisionItem key={x.decision_id} x={x} />)}
                </ul>
              </div>
            )
          })}
          {others.length > 0 && (
            <details className="border-t border-border pt-2">
              <summary className="cursor-pointer text-xs text-primary">Replaced and rejected proposals ({others.length})</summary>
              <ul className="mt-1 divide-y divide-border">
                {others.map((x) => <DecisionItem key={x.decision_id} x={x} />)}
              </ul>
            </details>
          )}
        </div>
      )}
    </Card>
  )
}

function DecisionItem({ x }: { x: PlanDecision }) {
  const look = DECISION_LOOK[x.status] ?? DECISION_LOOK.proposed
  const old = x.status === "superseded" || x.status === "rejected"
  return (
    <li className="flex gap-3 py-2">
      <span aria-hidden className={`mt-0.5 flex h-7 w-7 flex-none items-center justify-center rounded-full border text-xs font-bold ${look.cls}`}>
        {look.mark}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-xs text-muted">{x.label} · <span className="font-medium">{x.status_text}</span></p>
        <p className={`text-sm font-semibold ${old ? "text-muted line-through" : ""}`}>{x.value}</p>
        <p className="mt-0.5 text-xs text-muted">
          {x.proposed_by} proposed
          {x.agreed_by.length > 0 && ` · ${x.agreed_by.join(", ")} agreed`}
          {x.objected_by.length > 0 && ` · ${x.objected_by.join(", ")} objected`}
        </p>
        {x.rationale && (
          <p className="mt-1 border-l-2 border-primary/40 pl-2 text-xs italic">Why: {x.rationale}</p>
        )}
        {x.pending_reason && (
          <p className="mt-1 rounded-bar bg-status-warning/15 px-2 py-1 text-xs">Re-check: {x.pending_reason}</p>
        )}
        {x.timeline && x.timeline.length > 1 && (
          <details className="mt-1">
            <summary className="cursor-pointer text-xs text-primary">How it got here</summary>
            <ol className="mt-0.5 flex flex-col gap-0.5 border-l-2 border-border pl-2 text-xs text-muted">
              {x.timeline.map((t, i) => (
                <li key={i}>
                  {shortDate(t.time)} · {TIMELINE_WORD[t.what] ?? t.what}{t.by ? ` by ${t.by}` : ""}
                </li>
              ))}
            </ol>
          </details>
        )}
      </div>
    </li>
  )
}

const VERDICT: Record<string, { word: string; cls: string; bar: string }> = {
  feasible: { word: "Feasible", cls: "border-status-good text-status-good bg-status-good/10", bar: "bg-status-good" },
  risky: { word: "Risky", cls: "border-status-warning text-foreground bg-status-warning/10", bar: "bg-status-warning" },
  infeasible: { word: "Infeasible", cls: "border-status-critical text-status-critical bg-status-critical/10", bar: "bg-status-critical" },
}

function DependsOn({ deps }: { deps: PlanFeasibilityDependency[] }) {
  if (deps.length === 0) return null
  return (
    <div className="rounded-bar border border-status-warning bg-status-warning/10 px-2 py-1.5">
      <p className="text-xs font-semibold">It depends on something not settled yet</p>
      <ul className="mt-1 flex flex-col gap-1 text-xs">
        {deps.map((d) => (
          <li key={d.field}>
            {d.text.replace(/^whether the plan is feasible depends on /, "")
              .replace(/^./, (c) => c.toUpperCase())}.{" "}
            <span className="text-muted">{d.known === "unsure" ? "It was said unsurely; please confirm it." : "Nobody has said it yet."}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function FeasibilityCard({ f, deps = [] }: { f: PlanFeasibilityModel | null | undefined; deps?: PlanFeasibilityDependency[] }) {
  return (
    <Card
      title="Can the team finish this plan?"
      note="A trained model judges the plan from the task, algorithm, dataset size, GPU and time left. It learned from simulated project outcomes, so treat it as a second opinion, not a verdict."
    >
      {!f && deps.length > 0 ? (
        <DependsOn deps={deps} />
      ) : !f ? (
        <p className="text-sm text-muted">
          Not judged yet. Buddy needs the algorithm and the dataset size (for example &ldquo;about 10k rows&rdquo;).
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          <DependsOn deps={deps} />
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded-full border px-2 py-0.5 text-sm font-semibold ${(VERDICT[f.label] ?? VERDICT.risky).cls}`}>
              {(VERDICT[f.label] ?? VERDICT.risky).word}
            </span>
            <span className="text-xs text-muted">model confidence {Math.round(f.probability * 100)}%</span>
          </div>
          <ul className="flex flex-col gap-1" aria-label="Model probabilities">
            {Object.entries(f.probabilities).map(([label, p]) => (
              <li key={label} className="flex items-center gap-2 text-xs">
                <span className="w-20 text-muted">{(VERDICT[label] ?? { word: label }).word}</span>
                <span className="h-2 flex-1 overflow-hidden rounded-bar bg-chart-track">
                  <span className={`block h-full ${(VERDICT[label] ?? VERDICT.risky).bar}`} style={{ width: `${Math.round(p * 100)}%` }} />
                </span>
                <span className="w-9 text-right">{Math.round(p * 100)}%</span>
              </li>
            ))}
          </ul>
          {f.reasons.length > 0 && (
            <div>
              <p className="text-xs font-semibold">Main reasons</p>
              <ul className="mt-0.5 flex flex-col gap-0.5 text-xs text-muted">
                {f.reasons.map((r) => (
                  <li key={r.text}>
                    <span aria-hidden>{r.effect === "raises risk" ? "↑ " : "↓ "}</span>
                    {r.text} <span>({r.effect})</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {f.alternatives.length > 0 && (
            <div>
              <p className="text-xs font-semibold">Changes that would help</p>
              <ul className="mt-0.5 flex flex-col gap-0.5 text-xs">
                {f.alternatives.map((a) => (
                  <li key={a.change}>
                    {a.change.charAt(0).toUpperCase() + a.change.slice(1)}{" "}
                    <span className="text-muted">→ {Math.round(a.feasible_probability * 100)}% feasible</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="text-xs text-muted">{f.note}</p>
        </div>
      )}
    </Card>
  )
}

function PlanChecks({ checks }: { checks: PlanCheck[] }) {
  const problems = checks.filter((c) => c.status !== "pass").length
  return (
    <Card
      title="Plan checks"
      note="Buddy checks that the parts of the plan fit together (for example, that the metric suits the task). These are standard data science rules of thumb, not a verdict. To change the plan, agree on it in the chat."
    >
      {checks.length === 0 ? (
        <p className="text-sm text-muted">Nothing to check yet. Checks start once the task, algorithm or metric is known.</p>
      ) : (
        <div className="flex flex-col gap-2">
          <p className="text-sm">
            {problems === 0 ? "Everything checked so far fits together." : `${problems} ${problems === 1 ? "thing" : "things"} to look at.`}
          </p>
          <ul className="flex flex-col gap-1.5">
            {checks.map((c) => {
              const look = CHECK_LOOK[c.status] ?? CHECK_LOOK.warning
              return (
                <li key={c.key} className="rounded-bar border border-border px-2 py-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-semibold">{c.label}</span>
                    <span className={`rounded-full border px-2 py-0.5 text-xs ${look.cls}`}>
                      <span aria-hidden>{look.mark} </span>{look.word}
                    </span>
                  </div>
                  {c.detail && <p className="mt-1 text-xs text-muted">{c.detail}</p>}
                  {c.alternatives.length > 0 && c.status !== "pass" && (
                    <p className="mt-1 text-xs">
                      <span className="text-muted">Alternatives: </span>
                      {c.alternatives.map((a) => (
                        <span key={a} className="mr-1 inline-block rounded-bar border border-primary/40 bg-primary/5 px-1.5 py-0.5">
                          {a}
                        </span>
                      ))}
                    </p>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </Card>
  )
}

function TimelineDetails({ t, milestones }: { t: PlanTimeline | null | undefined; milestones: PlanMilestone[] }) {
  if (!t && milestones.length === 0) return null
  return (
    <div className="mt-1 flex flex-col gap-1 text-xs text-muted">
      {t?.deadline && (
        <p>
          {t.kind === "duration" ? "Ends about " : "Deadline "}
          <span className="font-semibold text-foreground">{dayText(t.deadline)}</span>
          {t.days_left !== null && (
            <span className={t.days_left < 0 ? "text-status-critical" : t.days_left <= 14 ? "text-status-warning" : ""}>
              {" "}· {daysLeftText(t.days_left)}
            </span>
          )}
        </p>
      )}
      {t?.kind === "week" && t.week !== null && <p>Teaching week {t.week} (no calendar date)</p>}
      {milestones.length > 0 && (
        <div>
          <p className="font-semibold text-foreground">Milestones</p>
          <ul className="mt-0.5 flex flex-col gap-0.5">
            {milestones.map((m) => (
              <li key={m.decision_id}>
                {m.name}: {m.date ? dayText(m.date) : m.week !== null ? `week ${m.week}` : m.when}
                <span> · from {m.proposed_by}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

const POLL_MS = 10000
const POLL_WHILE_READING_MS = 2500     // a document is being read: refresh sooner
const ACCEPT_TYPES = ".pdf,.docx,.txt,.md"

const STATUS: Record<string, { text: string; cls: string }> = {
  missing: { text: "Not discussed yet", cls: "border-border text-muted" },
  uncertain: { text: "Not sure yet", cls: "border-status-warning text-foreground bg-status-warning/10" },
  known: { text: "Mentioned", cls: "border-primary text-primary bg-primary/10" },
  confirmed: { text: "Confirmed", cls: "border-status-good text-status-good bg-status-good/10" },
  conflicting: { text: "Team disagrees", cls: "border-status-critical text-status-critical bg-status-critical/10" },
}

// The coloured stripe down a field card's left edge: its status at a glance.
const STRIPE: Record<string, string> = {
  missing: "border-l-border",
  uncertain: "border-l-status-warning",
  known: "border-l-primary",
  confirmed: "border-l-status-good",
  conflicting: "border-l-status-critical",
}

// The fields, grouped by the part of the plan they describe.
const FIELD_GROUPS: { title: string; fields: string[] }[] = [
  { title: "The project", fields: ["domain", "problem", "objective"] },
  { title: "The data", fields: ["dataset", "dataset_size", "target_variable"] },
  { title: "The model", fields: ["task_type", "algorithm", "evaluation_metric"] },
  { title: "Practical limits", fields: ["resources", "timeline"] },
]

type Decide = (s: PlanSuggestion, action: "accept" | "dismiss") => void

function StatusBadge({ status }: { status: string }) {
  const s = STATUS[status] ?? { text: status, cls: "border-border text-muted" }
  return <span className={`rounded-full border px-2 py-0.5 text-xs ${s.cls}`}>{s.text}</span>
}

const STATUS_WORD: Record<string, string> = {
  missing: "not discussed", uncertain: "not sure", known: "mentioned", confirmed: "confirmed", conflicting: "disagreement",
}

function HistoryList({ entries, showField = false }: { entries: PlanHistoryEntry[]; showField?: boolean }) {
  if (entries.length === 0) return <p className="text-xs text-muted">No changes yet.</p>
  return (
    <ol className="flex flex-col gap-1.5 border-l-2 border-border pl-3">
      {entries.map((e) => (
        <li key={e.version} className="relative text-xs">
          <span className="absolute -left-[17px] top-1 h-2 w-2 rounded-full bg-primary" aria-hidden />
          <p>
            {showField && <span className="font-semibold">{e.label}: </span>}
            <span className="text-foreground">{e.summary}</span>
          </p>
          <p className="text-muted">
            {shortDate(e.time)}
            {e.status_before !== e.status_after &&
              ` · ${STATUS_WORD[e.status_before] ?? e.status_before} → ${STATUS_WORD[e.status_after] ?? e.status_after}`}
            {e.confidence_band ? ` · confidence ${e.confidence_band}` : ""}
          </p>
          {e.text && <p className="mt-0.5 text-muted">&ldquo;{e.text}&rdquo;</p>}
        </li>
      ))}
    </ol>
  )
}

function FieldHistory({ teamId, field, version }: { teamId: string; field: string; version: number }) {
  const [open, setOpen] = useState(false)
  const [entries, setEntries] = useState<PlanHistoryEntry[] | null>(null)
  const [error, setError] = useState("")

  useEffect(() => {
    if (!open) return
    let alive = true
    getPlanHistory(teamId, field)
      .then((r) => { if (alive) { setEntries(r.entries); setError("") } })
      .catch((e) => alive && setError(e.message))
    return () => { alive = false }
  }, [open, teamId, field, version])          // reload while open whenever the blueprint changes

  return (
    <details className="mt-1" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className="cursor-pointer text-xs text-primary">History</summary>
      <div className="mt-1">
        {error ? <p className="text-xs text-status-critical">{error}</p>
          : entries === null ? <p className="text-xs text-muted">Loading...</p>
          : <HistoryList entries={entries} />}
      </div>
    </details>
  )
}

function Suggestions({ items, canEdit, busy, onDecide }: {
  items: PlanSuggestion[]
  canEdit: boolean
  busy: string | null
  onDecide: Decide
}) {
  return (
    <div className="mt-2 rounded-bar border border-dashed border-primary/40 bg-primary/5 px-2 py-1.5">
      <p className="text-xs font-semibold text-primary">Found in your documents</p>
      <ul className="mt-1 flex flex-col gap-1.5">
        {items.map((s) => (
          <li key={s.fact_id} className="flex flex-col gap-1">
            <p className="text-sm">
              {s.value}{" "}
              <span className="text-xs text-muted">
                · {s.source}
                {s.mentions > 1 ? ` · mentioned ${s.mentions} times` : ""}
              </span>
            </p>
            {s.text && <p className="border-l-2 border-border pl-2 text-xs text-muted">&ldquo;{s.text}&rdquo;</p>}
            {canEdit && (
              <div className="flex gap-2">
                <Button variant="secondary" className="px-2 py-1 text-xs" disabled={busy !== null}
                        aria-label={`Use ${s.value}`} onClick={() => onDecide(s, "accept")}>
                  {busy === s.fact_id ? "Saving..." : "Use this"}
                </Button>
                <Button variant="ghost" className="px-2 py-1 text-xs" disabled={busy !== null}
                        aria-label={`Dismiss ${s.value}`} onClick={() => onDecide(s, "dismiss")}>
                  Dismiss
                </Button>
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}

function FieldCard({ f, teamId, version, milestones, showSources, canEdit, busy, onDecide }: {
  f: PlanField
  teamId: string
  version: number
  milestones: PlanMilestone[]
  showSources: boolean
  canEdit: boolean
  busy: string | null
  onDecide: Decide
}) {
  return (
    <li className={`rounded-card border border-l-4 border-border ${STRIPE[f.status] ?? "border-l-border"} bg-surface px-3 py-2`}>
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">{f.label}</h3>
        <StatusBadge status={f.status} />
      </div>

      {f.status === "conflicting" && f.candidates?.length ? (
        <div className="mt-1">
          <p className="text-xs text-muted">Different answers so far:</p>
          <ul className="mt-1 flex flex-wrap gap-1">
            {f.candidates.map((c) => (
              <li key={c} className="rounded-full border border-status-critical/40 bg-status-critical/5 px-2 py-0.5 text-sm">{c}</li>
            ))}
          </ul>
        </div>
      ) : f.value ? (
        <p className="mt-1 text-base font-medium">{f.value}</p>
      ) : (
        <p className="mt-1 text-sm text-subtle">Nothing recorded yet</p>
      )}

      {f.confidence_band && (
        <p className="mt-1 text-xs text-muted">
          Confidence: {f.confidence_band} · mentioned {f.mentions} {f.mentions === 1 ? "time" : "times"}
        </p>
      )}

      {(f.at_risk || f.needs_detail) && (
        <div className="mt-1.5 flex flex-col gap-1">
          {f.at_risk && (
            <p className="text-xs">
              <span className="mr-1 rounded-full bg-status-warning/25 px-1.5 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide">May change</span>
              <span className="text-muted">{f.at_risk.length ? f.at_risk.join("; ") : "it may still change"}</span>
            </p>
          )}
          {f.needs_detail && (
            <p className="text-xs">
              <span className="mr-1 rounded-full bg-primary/15 px-1.5 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide text-primary">Needs detail</span>
              <span className="text-muted">{f.needs_detail}</span>
            </p>
          )}
        </div>
      )}

      {f.name === "timeline" && <TimelineDetails t={f.timeline} milestones={milestones} />}

      {f.suggestions && f.suggestions.length > 0 && (
        <Suggestions items={f.suggestions} canEdit={canEdit} busy={busy} onDecide={onDecide} />
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

      {f.mentions > 0 && <FieldHistory teamId={teamId} field={f.name} version={version} />}
    </li>
  )
}

function documentStatus(d: PlanDocument) {
  if (d.status === "processing") return "Reading..."
  if (d.status === "failed") return d.error || "Could not be read"
  const n = d.facts ?? 0
  return n === 0 ? "No planning details found" : `${n} planning ${n === 1 ? "detail" : "details"} found`
}

function DocumentsCard({ teamId, documents, canEdit, onChanged }: {
  teamId: string
  documents: PlanDocument[]
  canEdit: boolean
  onChanged: () => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState("")

  const upload = async (file: File | undefined) => {
    if (!file) return
    setUploading(true)
    setError("")
    try {
      await uploadPlanDocument(teamId, file)
      onChanged()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setUploading(false)
      if (input.current) input.current.value = ""
    }
  }

  const remove = async (d: PlanDocument) => {
    if (!window.confirm(`Remove ${d.filename}? Its suggestions that nobody accepted are removed too.`)) return
    setError("")
    try {
      await removePlanDocument(teamId, d.doc_id)
      onChanged()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <Card
      title="Project documents"
      note="Upload your proposal or requirement documents (PDF, Word, text or Markdown, up to 10 MB). Buddy reads them and suggests values for the blueprint; nothing is used until someone in the team accepts it."
    >
      <div className="flex flex-col gap-2">
        {canEdit && (
          <div className="flex flex-wrap items-center gap-2">
            <input ref={input} type="file" accept={ACCEPT_TYPES} className="sr-only" id="c1-doc-upload"
                   onChange={(e) => upload(e.target.files?.[0])} disabled={uploading} />
            <Button variant="secondary" disabled={uploading} onClick={() => input.current?.click()}>
              {uploading ? "Uploading..." : "Upload a document"}
            </Button>
            <span className="text-xs text-muted">Only the text is kept, not the file.</span>
          </div>
        )}
        {error && <Notice tone="error">{error}</Notice>}
        {documents.length === 0 ? (
          <p className="text-sm text-muted">No documents yet.</p>
        ) : (
          <ul className="flex flex-col gap-1">
            {documents.map((d) => (
              <li key={d.doc_id} className="flex items-center justify-between gap-2 rounded-bar border border-border px-2 py-1">
                <div className="min-w-0">
                  <p className="truncate text-sm">{d.filename}</p>
                  <p className={`text-xs ${d.status === "failed" ? "text-status-critical" : "text-muted"}`}>
                    {documentStatus(d)} · {shortDate(d.uploaded_at)}
                  </p>
                </div>
                {canEdit && (
                  <Button variant="ghost" className="px-2 py-1 text-xs" aria-label={`Remove ${d.filename}`}
                          onClick={() => remove(d)}>
                    Remove
                  </Button>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  )
}

export function BlueprintPlan({ teamId }: { teamId: string }) {
  const [plan, setPlan] = useState<PlanData | null>(null)
  const [unavailable, setUnavailable] = useState("")
  const [error, setError] = useState("")
  const [busy, setBusy] = useState<string | null>(null)
  const [actionError, setActionError] = useState("")
  const [section, setSection] = useState<Section>("fields")
  const alive = useRef(true)

  const load = useCallback(
    () =>
      getPlan(teamId)
        .then((p) => {
          if (!alive.current) return
          if (p.available) { setPlan(p); setUnavailable("") }
          else setUnavailable(p.reason)
          setError("")
        })
        .catch((e) => alive.current && setError(e.message)),
    [teamId],
  )

  useEffect(() => {
    alive.current = true
    setPlan(null)
    setUnavailable("")
    setError("")
    load()
    return () => { alive.current = false }
  }, [load])

  const reading = plan?.documents?.some((d) => d.status === "processing") ?? false
  useEffect(() => {
    const id = setInterval(load, reading ? POLL_WHILE_READING_MS : POLL_MS)
    return () => clearInterval(id)
  }, [load, reading])

  const decide: Decide = async (s, action) => {
    setBusy(s.fact_id)
    setActionError("")
    try {
      const res = await decideSuggestion(teamId, s.fact_id, action)
      setPlan((old) => (old ? { ...old, ...res.plan } : old))
    } catch (e) {
      setActionError((e as Error).message)
      load()
    } finally {
      setBusy(null)
    }
  }

  if (unavailable) return null   // research teams: only the old blueprint card is shown
  if (error) return <Notice tone="error">{error}</Notice>
  if (!plan) return <p className="text-sm text-muted">Loading the planning blueprint...</p>

  const r = plan.readiness
  const pct = r.total ? Math.round((r.confirmed / r.total) * 100) : 0
  const canEdit = plan.can_edit ?? false
  const recheck = plan.decisions?.needs_recheck.length ?? 0
  const problems = (plan.checks ?? []).filter((c) => c.status !== "pass").length
  const verdict = plan.feasibility_model ? VERDICT[plan.feasibility_model.label] : undefined
  const openSuggestions = plan.fields.reduce((n, f) => n + (f.suggestions?.length ?? 0), 0)

  const tabs: { key: Section; label: string; count?: number; alert?: boolean }[] = [
    { key: "fields", label: "Fields", count: r.total, alert: r.conflicting.length > 0 },
    ...(plan.decisions ? [{ key: "decisions" as Section, label: "Decisions", count: plan.decisions.current.length, alert: recheck > 0 }] : []),
    ...(plan.checks || "feasibility_model" in plan ? [{ key: "checks" as Section, label: "Plan check", count: problems || undefined, alert: problems > 0 }] : []),
    ...(plan.documents ? [{ key: "documents" as Section, label: "Documents", count: plan.documents.length }] : []),
    { key: "history", label: "History" },
  ]
  const shown: Section = tabs.some((t) => t.key === section) ? section : "fields"
  const fieldsByName = new Map(plan.fields.map((f) => [f.name, f]))
  const grouped = FIELD_GROUPS.map((g) => ({ ...g, items: g.fields.flatMap((n) => fieldsByName.get(n) ?? []) }))
  const ungrouped = plan.fields.filter((f) => !FIELD_GROUPS.some((g) => g.fields.includes(f.name)))
  if (ungrouped.length) grouped.push({ title: "Other", fields: [], items: ungrouped })

  return (
    <div className="flex flex-col gap-3">
      <Card
        title="Planning blueprint"
        note="Built by Buddy from your team chat and uploaded documents. Nothing here is final until the team confirms it."
      >
        <div className="flex flex-col gap-3">
          <div>
            <p className="text-sm">
              <span className="font-semibold">{r.confirmed} of {r.total}</span> fields confirmed
              {r.complete ? " · ready to build" : ""}
            </p>
            <div className="mt-1 h-2 w-full overflow-hidden rounded-bar bg-chart-track" role="progressbar"
                 aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
              <div className="h-full bg-primary" style={{ width: `${pct}%` }} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <SummaryTile label="Disagreements" value={String(r.conflicting.length)}
                         tone={r.conflicting.length ? "critical" : "neutral"} onClick={() => setSection("fields")} />
            {plan.decisions && (
              <SummaryTile label="Decisions" value={String(plan.decisions.current.length)}
                           sub={recheck ? `${recheck} need${recheck === 1 ? "s" : ""} re-check` : undefined}
                           tone={recheck ? "warning" : "neutral"} onClick={() => setSection("decisions")} />
            )}
            {(plan.checks || "feasibility_model" in plan) && (
              <SummaryTile label="Plan" value={verdict?.word ?? "Not judged"}
                           sub={problems ? `${problems} check${problems === 1 ? "" : "s"} to look at` : undefined}
                           tone={plan.feasibility_model?.label === "infeasible" ? "critical"
                             : plan.feasibility_model?.label === "risky" || problems ? "warning" : "neutral"}
                           onClick={() => setSection("checks")} />
            )}
            {plan.documents && (
              <SummaryTile label="Document suggestions" value={String(openSuggestions)}
                           tone="neutral" onClick={() => setSection("fields")} />
            )}
          </div>
          {plan.next_questions && plan.next_questions.length > 0 ? (
            <div>
              <p className="text-xs font-semibold">What to settle next</p>
              <ol className="mt-1 flex flex-col gap-1">
                {plan.next_questions.map((q, i) => (
                  <li key={`${q.kind}-${q.field}`} className="flex gap-2 text-xs">
                    <span aria-hidden className="flex h-5 w-5 flex-none items-center justify-center rounded-full bg-primary text-[0.65rem] font-semibold text-primary-foreground">
                      {i + 1}
                    </span>
                    <span className="min-w-0">
                      <span className="font-semibold">
                        {q.action.charAt(0).toUpperCase() + q.action.slice(1)} the {q.label.toLowerCase()}
                      </span>
                      {q.value ? <span className="text-muted"> ({q.value})</span> : null}
                      <span className="block text-muted">{q.why.charAt(0).toUpperCase() + q.why.slice(1)}</span>
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          ) : plan.next_gaps.length > 0 && (
            <p className="text-xs text-muted">
              Buddy will ask about next: {plan.next_gaps.join(", ").replaceAll("_", " ")}.
            </p>
          )}
          {plan.updated_at && <p className="text-xs text-subtle">Last change {shortDate(plan.updated_at)}</p>}
        </div>
      </Card>

      <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Blueprint sections">
        {tabs.map((t) => (
          <button key={t.key} type="button" role="tab" aria-selected={shown === t.key} onClick={() => setSection(t.key)}
                  className={`inline-flex items-center gap-1 rounded-full border px-3 py-1.5 text-xs font-medium ${
                    shown === t.key ? "border-primary bg-primary text-primary-foreground" : "border-border bg-surface"}`}>
            {t.label}
            {t.count !== undefined && <span className="opacity-75">{t.count}</span>}
            {t.alert && <span className="h-1.5 w-1.5 rounded-full bg-status-critical" aria-label="needs attention" />}
          </button>
        ))}
      </div>

      {actionError && <Notice tone="error">{actionError}</Notice>}

      <div role="tabpanel" className="flex flex-col gap-3">
        {shown === "fields" && (
          <>
            {grouped.map((g) => g.items.length > 0 && (
              <section key={g.title} className="flex flex-col gap-1.5">
                <h2 className="text-sm font-semibold">{g.title}</h2>
                <ul className="flex flex-col gap-2">
                  {g.items.map((f) => (
                    <FieldCard key={f.name} f={f} teamId={teamId} version={plan.version} milestones={plan.milestones ?? []}
                               showSources={plan.can_see_sources}
                               canEdit={canEdit} busy={busy} onDecide={decide} />
                  ))}
                </ul>
              </section>
            ))}
            {!plan.can_see_sources && (
              <p className="text-xs text-muted">
                You can see values and status only. The quoted chat and document words are private to the team lead and members.
              </p>
            )}
          </>
        )}

        {shown === "decisions" && plan.decisions && <DecisionsCard d={plan.decisions} />}

        {shown === "checks" && (
          <>
            {"feasibility_model" in plan && <FeasibilityCard f={plan.feasibility_model} deps={plan.feasibility_depends_on} />}
            {plan.checks && <PlanChecks checks={plan.checks} />}
          </>
        )}

        {shown === "documents" && plan.documents && (
          <>
            <DocumentsCard teamId={teamId} documents={plan.documents} canEdit={canEdit} onChanged={load} />
            {openSuggestions > 0 && (
              <p className="text-xs text-muted">
                Suggestions from your documents are shown on the matching field in the Fields tab.
              </p>
            )}
          </>
        )}

        {shown === "history" && (
          <Card title="Recent changes" note="How the plan has changed lately. Open a field's History to see all of its changes.">
            {plan.recent_changes && plan.recent_changes.length > 0
              ? <HistoryList entries={plan.recent_changes} showField />
              : <p className="text-sm text-muted">No changes yet.</p>}
          </Card>
        )}
      </div>
    </div>
  )
}

type Section = "fields" | "decisions" | "checks" | "documents" | "history"

const TILE_TONE: Record<string, string> = {
  neutral: "border-border",
  warning: "border-status-warning bg-status-warning/10",
  critical: "border-status-critical/60 bg-status-critical/5",
}

function SummaryTile({ label, value, sub, tone, onClick }: {
  label: string
  value: string
  sub?: string
  tone: "neutral" | "warning" | "critical"
  onClick: () => void
}) {
  return (
    <button type="button" onClick={onClick}
            className={`rounded-bar border px-2 py-1.5 text-left focus-visible:outline-2 focus-visible:outline-primary ${TILE_TONE[tone]}`}>
      <span className="block text-xs text-muted">{label}</span>
      <span className="block text-base font-semibold">{value}</span>
      {sub && <span className="block text-xs text-muted">{sub}</span>}
    </button>
  )
}
