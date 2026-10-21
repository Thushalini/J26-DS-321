"use client"
// The nine-field planning blueprint built by Component 1 (Buddy) from the team chat and from
// uploaded project documents. Values found in documents are suggestions until the team accepts them.
import { useCallback, useEffect, useRef, useState } from "react"

import { Button, Card, Notice } from "@/components/ui"
import {
  decideSuggestion,
  getPlan,
  removePlanDocument,
  shortDate,
  uploadPlanDocument,
  type PlanData,
  type PlanDocument,
  type PlanField,
  type PlanSuggestion,
} from "@/lib/api"

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

type Decide = (s: PlanSuggestion, action: "accept" | "dismiss") => void

function StatusBadge({ status }: { status: string }) {
  const s = STATUS[status] ?? { text: status, cls: "border-border text-muted" }
  return <span className={`rounded-full border px-2 py-0.5 text-xs ${s.cls}`}>{s.text}</span>
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

function FieldCard({ f, showSources, canEdit, busy, onDecide }: {
  f: PlanField
  showSources: boolean
  canEdit: boolean
  busy: string | null
  onDecide: Decide
}) {
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

  return (
    <div className="flex flex-col gap-3">
      <Card
        title="Planning blueprint (built by Buddy from your chat and documents)"
        note="Buddy listens to the team chat and reads your uploaded documents to fill these nine fields. Nothing here is final until the team confirms it."
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

      {plan.documents && (
        <DocumentsCard teamId={teamId} documents={plan.documents} canEdit={canEdit} onChanged={load} />
      )}

      {actionError && <Notice tone="error">{actionError}</Notice>}

      <ul className="flex flex-col gap-2">
        {plan.fields.map((f) => (
          <FieldCard key={f.name} f={f} showSources={plan.can_see_sources} canEdit={canEdit}
                     busy={busy} onDecide={decide} />
        ))}
      </ul>

      {!plan.can_see_sources && (
        <p className="text-xs text-muted">
          You can see values and status only. The quoted chat and document words are private to the team lead and members.
        </p>
      )}
      <p className="text-xs text-muted">{plan.note}</p>
    </div>
  )
}
