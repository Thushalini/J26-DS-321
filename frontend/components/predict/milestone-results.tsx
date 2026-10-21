"use client"

import { useCallback, useEffect, useState } from "react"

import { SectionCard } from "@/components/section-card"
import { Button, Notice, inputClass } from "@/components/ui"
import { formatDate } from "@/lib/format"
import { api } from "@/lib/api"
import { predictCopy } from "@/lib/predict/config"
import type { MilestoneResult } from "@/lib/predict/types"

const copy = predictCopy.milestones
const OPTIONS = Object.keys(copy.options) as (keyof typeof copy.options)[]

export function MilestoneResults({ teamId, canRecord }: { teamId: string; canRecord: boolean }) {
  const base = `/c4/teams/${encodeURIComponent(teamId)}/milestone-results`
  const [results, setResults] = useState<MilestoneResult[]>([])
  const [milestone, setMilestone] = useState("")
  const [date, setDate] = useState("")
  const [result, setResult] = useState<string>(OPTIONS[0])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  const act = useCallback(async (fn: () => Promise<MilestoneResult[]>) => {
    setBusy(true)
    setError("")
    try {
      setResults(await fn())
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    act(() => api<MilestoneResult[]>(base))
  }, [act, base])

  const save = () => act(async () => {
    const listed = await api<MilestoneResult[]>(base, { method: "POST", json: { milestone, milestone_date: date, result } })
    setMilestone("")
    setDate("")
    return listed
  })

  const remove = (id: string) => act(() => api<MilestoneResult[]>(`${base}/${encodeURIComponent(id)}`, { method: "DELETE" }))

  return (
    <SectionCard title={copy.title} description={copy.description}>
      <div className="flex flex-col gap-3">
        {results.length === 0 && <p className="text-sm text-muted">{copy.empty}</p>}
        {results.length > 0 && (
          <ul className="flex flex-col divide-y divide-border">
            {results.map((r) => (
              <li key={r.result_id} className="flex items-center justify-between gap-2 py-2 text-sm">
                <span className="min-w-0">
                  <span className="block truncate font-medium">{r.milestone}</span>
                  <span className="block text-xs text-muted">
                    {formatDate(r.date)} · {copy.options[r.result as keyof typeof copy.options] ?? r.result}
                  </span>
                </span>
                {canRecord && (
                  <Button variant="ghost" className="px-1 py-0 text-xs text-danger" disabled={busy}
                          onClick={() => remove(r.result_id)}>{copy.remove}</Button>
                )}
              </li>
            ))}
          </ul>
        )}

        {canRecord ? (
          <form className="flex flex-col gap-2" onSubmit={(e) => { e.preventDefault(); save() }}>
            <label className="text-xs text-muted">
              {copy.name}
              <input id="milestone-name" className={`${inputClass} mt-1`} value={milestone} required
                     placeholder={copy.namePlaceholder} onChange={(e) => setMilestone(e.target.value)} />
            </label>
            <div className="flex flex-wrap gap-2">
              <label className="min-w-0 flex-1 text-xs text-muted">
                {copy.date}
                <input id="milestone-date" type="date" className={`${inputClass} mt-1`} value={date} required
                       onChange={(e) => setDate(e.target.value)} />
              </label>
              <label className="min-w-0 flex-1 text-xs text-muted">
                {copy.result}
                <select id="milestone-result" className={`${inputClass} mt-1`} value={result}
                        onChange={(e) => setResult(e.target.value)}>
                  {OPTIONS.map((o) => <option key={o} value={o}>{copy.options[o]}</option>)}
                </select>
              </label>
            </div>
            <div>
              <Button type="submit" disabled={busy || !milestone.trim() || !date}>{busy ? copy.saving : copy.save}</Button>
            </div>
          </form>
        ) : (
          <p className="text-xs text-muted">{copy.readOnly}</p>
        )}

        {error && <Notice tone="error">{error}</Notice>}
      </div>
    </SectionCard>
  )
}
