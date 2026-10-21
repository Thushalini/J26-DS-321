"use client"

import { useCallback, useEffect, useState } from "react"

import { SectionCard } from "@/components/section-card"
import { Button, Notice } from "@/components/ui"
import { api, shortDate } from "@/lib/api"
import { predictCopy } from "@/lib/predict/config"
import type { HelperSuggestion, SavedSheet, WeakStage } from "@/lib/predict/types"

const copy = predictCopy.helpers
const STAGES = Object.keys(copy.stages) as (keyof typeof copy.stages)[]

function asBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] ?? "")
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

export function HelperSuggestions({ teamId, weakStages }: { teamId: string; weakStages: WeakStage[] }) {
  const base = `/c4/teams/${encodeURIComponent(teamId)}`
  const [sheets, setSheets] = useState<SavedSheet[]>([])
  const [stages, setStages] = useState<Set<string>>(new Set())
  const [result, setResult] = useState<HelperSuggestion[] | null>(null)
  const [error, setError] = useState("")
  const [busy, setBusy] = useState<string | null>(null)

  const sources = Object.fromEntries(weakStages.map((w) => [w.stage, w.sources]))

  useEffect(() => {
    setStages(new Set(weakStages.map((w) => w.stage)))
  }, [weakStages])

  const act = useCallback(async (key: string, fn: () => Promise<void>) => {
    setBusy(key)
    setError("")
    try {
      await fn()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }, [])

  useEffect(() => {
    act("load", async () => setSheets(await api<SavedSheet[]>(`${base}/sheets`)))
  }, [act, base])

  const upload = (userId: string, file: File | undefined) => file && act(userId, async () => {
    setSheets(await api<SavedSheet[]>(`${base}/sheets`, {
      method: "POST", json: { member_user_id: userId, content_base64: await asBase64(file) },
    }))
  })

  const remove = (userId: string) => act(userId, async () => {
    setSheets(await api<SavedSheet[]>(`${base}/sheets/${encodeURIComponent(userId)}`, { method: "DELETE" }))
  })

  const toggle = (stage: string) => {
    const next = new Set(stages)
    if (next.has(stage)) next.delete(stage)
    else next.add(stage)
    setStages(next)
  }

  const run = () => act("run", async () => {
    setResult(null)
    const body = await api<{ suggestions: HelperSuggestion[] }>(`${base}/helpers`, {
      method: "POST", json: { stages: Array.from(stages) },
    })
    setResult(body.suggestions)
  })

  const savedCount = sheets.filter((s) => s.saved).length

  return (
    <SectionCard title={copy.title} description={copy.description}>
      <div className="flex flex-col gap-4">
        <p className="text-xs text-muted">{copy.privacy}</p>

        <ul className="flex flex-col gap-2">
          {sheets.map((m) => (
            <li key={m.user_id} className="flex flex-col gap-1 rounded-card border border-border p-2 text-sm">
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">{m.name}</span>
                <span className="flex items-center gap-2 text-xs">
                  <span className={m.saved ? "text-success" : "text-muted"}>
                    {busy === m.user_id ? copy.uploading : m.saved ? `${copy.saved} · ${shortDate(m.uploaded_at)}` : copy.notSaved}
                  </span>
                  {m.saved && (
                    <Button variant="ghost" className="px-1 py-0 text-xs text-danger" disabled={busy !== null}
                            onClick={() => remove(m.user_id)}>{copy.remove}</Button>
                  )}
                </span>
              </div>
              <label className="text-xs text-muted">
                {copy.memberFile}
                <input type="file" accept="application/pdf" className="mt-1 block w-full text-xs" disabled={busy !== null}
                       onChange={(e) => { upload(m.user_id, e.target.files?.[0]); e.target.value = "" }} />
              </label>
            </li>
          ))}
        </ul>

        <div>
          <p className="text-sm font-medium">{copy.stagesTitle}</p>
          <p className="text-xs text-muted">{weakStages.length ? copy.stagesDetected : copy.stagesNone}</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {STAGES.map((stage) => (
              <label key={stage} className="flex items-center gap-1.5 rounded-full border border-border px-2 py-1 text-xs">
                <input type="checkbox" checked={stages.has(stage)} onChange={() => toggle(stage)} />
                {copy.stages[stage]}
                {sources[stage] && (
                  <span className="text-muted">
                    ({copy.sourcePrefix} {sources[stage].map((s) => copy.sources[s]).join(", ")})
                  </span>
                )}
              </label>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <div>
            <Button onClick={run} disabled={busy !== null || savedCount === 0 || stages.size === 0}>
              {busy === "run" ? copy.running : copy.run}
            </Button>
          </div>
          {savedCount === 0 && <p className="text-xs text-muted">{copy.needSheets}</p>}
        </div>

        {error && <Notice tone="error">{error}</Notice>}

        {result && (
          <div>
            <p className="text-sm font-semibold">{copy.resultTitle}</p>
            <ul className="mt-2 flex flex-col divide-y divide-border">
              {result.map((s) => (
                <li key={s.stage} className="py-2 text-sm">
                  <span className="font-medium">{s.stage_label}</span>
                  {" → "}
                  <span className={s.member_name ? "font-semibold text-primary" : "text-muted"}>{s.member_name ?? copy.noMatch}</span>
                  <p className="text-xs text-muted">{s.reason}</p>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </SectionCard>
  )
}
