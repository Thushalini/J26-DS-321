"use client"
// My profile (private to the signed-in person): about me, my teams, strengths and growth areas
// worked out from my activity (each with its evidence), my work mix, and my own private notes.
// Data: GET /app/me/profile and PUT /app/me/notes (backend/buddy/profile.py).
import type React from "react"
import { useEffect, useState } from "react"

import { Button, Card, Field, Notice, Pill, inputClass } from "@/components/ui"
import { api, ROLE_LABEL, shortDate } from "@/lib/api"

interface ProfileItem { title: string; evidence: string; source: string }
interface ProfileNotes { strengths: string | null; improve: string | null; goals: string | null; updated_at: string | null }
interface ProfileTeam {
  team_id: string
  name: string
  team_role: string
  member_code: string | null
  git_identities: string[]
  joined_at: string | null
}
interface Profile {
  about: { user_id: string; email: string; name: string; role: string; joined: string | null }
  teams: ProfileTeam[]
  notes: ProfileNotes
  has_activity: boolean
  work_mix: { mix: Record<string, number>; code_items: number; chat_items: number } | null
  strengths: ProfileItem[]
  growth: ProfileItem[]
}

const STAGE_LABEL: Record<string, string> = {
  data_cleaning: "Data cleaning",
  eda: "Exploratory analysis",
  feature_engineering: "Feature engineering",
  model_development: "Model development",
  model_evaluation: "Model evaluation",
  deployment_mlops: "Deployment / MLOps",
  documentation: "Documentation",
}

function dateOnly(ts: string | null) {
  return ts ? new Date(ts).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "unknown"
}

function initials(name: string) {
  return name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]?.toUpperCase()).join("")
}

function ItemList({ items, tone, empty }: { items: ProfileItem[]; tone: "good" | "grow"; empty: string }) {
  if (items.length === 0) return <p className="text-sm text-muted">{empty}</p>
  const mark = tone === "good"
    ? { sign: "✓", cls: "border-status-good text-status-good bg-status-good/10" }
    : { sign: "↗", cls: "border-status-warning text-foreground bg-status-warning/15" }
  return (
    <ul className="divide-y divide-border">
      {items.map((it) => (
        <li key={it.title} className="flex gap-3 py-2">
          <span aria-hidden className={`mt-0.5 flex h-7 w-7 flex-none items-center justify-center rounded-full border text-xs font-bold ${mark.cls}`}>
            {mark.sign}
          </span>
          <div className="min-w-0">
            <p className="text-sm font-semibold">{it.title}</p>
            <p className="text-xs text-muted">{it.evidence}</p>
            <span className="mt-1 inline-block"><Pill>{it.source}</Pill></span>
          </div>
        </li>
      ))}
    </ul>
  )
}

function WorkMix({ w }: { w: NonNullable<Profile["work_mix"]> }) {
  const rows = Object.entries(w.mix).sort((a, b) => b[1] - a[1])
  return (
    <Card title="My work mix" note={`What your ${w.code_items} code items and ${w.chat_items} chat messages were about, from the Behavior tab.`}>
      <ul className="flex flex-col gap-1.5" aria-label="Share of work per project stage">
        {rows.map(([stage, share]) => (
          <li key={stage} className="flex items-center gap-2 text-xs">
            <span className="w-32 flex-none text-muted">{STAGE_LABEL[stage] ?? stage}</span>
            <span className="h-2 flex-1 overflow-hidden rounded-bar bg-chart-track">
              <span className="block h-full bg-primary" style={{ width: `${Math.round(share * 100)}%` }} />
            </span>
            <span className="w-9 text-right">{Math.round(share * 100)}%</span>
          </li>
        ))}
      </ul>
    </Card>
  )
}

function NotesCard({ notes, onSaved }: { notes: ProfileNotes; onSaved: (n: ProfileNotes) => void }) {
  const [form, setForm] = useState({ strengths: notes.strengths ?? "", improve: notes.improve ?? "", goals: notes.goals ?? "" })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const [done, setDone] = useState(false)
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setForm({ ...form, [k]: e.target.value })
    setDone(false)
  }

  const save = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError("")
    try {
      onSaved(await api<ProfileNotes>("/app/me/notes", { method: "PUT", json: form }))
      setDone(true)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card title="My notes" note="Your own words about yourself. Only you can see them.">
      <form className="flex flex-col gap-3" onSubmit={save}>
        <Field label="My strengths">
          <textarea className={inputClass} rows={3} maxLength={2000} value={form.strengths} onChange={set("strengths")}
                    placeholder="For example: good at cleaning messy data" />
        </Field>
        <Field label="Things I want to improve">
          <textarea className={inputClass} rows={3} maxLength={2000} value={form.improve} onChange={set("improve")}
                    placeholder="For example: writing tests for my code" />
        </Field>
        <Field label="My goals for this project">
          <textarea className={inputClass} rows={3} maxLength={2000} value={form.goals} onChange={set("goals")}
                    placeholder="For example: deploy the model myself" />
        </Field>
        {error && <Notice tone="error">{error}</Notice>}
        {done && <Notice tone="success">Your notes are saved.</Notice>}
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-subtle">{notes.updated_at ? `Last saved ${shortDate(notes.updated_at)}` : "Not saved yet"}</span>
          <Button type="submit" disabled={busy}>{busy ? "Saving..." : "Save notes"}</Button>
        </div>
      </form>
    </Card>
  )
}

export function MyProfile() {
  const [p, setP] = useState<Profile | null>(null)
  const [error, setError] = useState("")

  useEffect(() => {
    let alive = true
    api<Profile>("/app/me/profile")
      .then((r) => { if (alive) setP(r) })
      .catch((e) => alive && setError(e.message))
    return () => { alive = false }
  }, [])

  if (error) return <Notice tone="error">{error}</Notice>
  if (!p) return <p className="text-sm text-muted">Loading your profile...</p>

  const student = p.teams.some((t) => t.team_role === "lead" || t.team_role === "member")

  return (
    <>
      <Card>
        <div className="flex items-center gap-3">
          <span aria-hidden className="flex h-12 w-12 flex-none items-center justify-center rounded-full bg-primary text-base font-semibold text-primary-foreground">
            {initials(p.about.name)}
          </span>
          <div className="min-w-0">
            <h1 className="truncate text-lg font-semibold">{p.about.name}</h1>
            <p className="truncate text-sm text-muted">{p.about.email}</p>
            <p className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted">
              <Pill tone="primary">{ROLE_LABEL[p.about.role] ?? p.about.role}</Pill>
              <span>Joined {dateOnly(p.about.joined)}</span>
            </p>
          </div>
        </div>
        <p className="mt-3 text-xs text-subtle">Only you can see this page. Teammates and supervisors cannot open it.</p>
      </Card>

      <Card title={student ? "My teams" : "My groups"}>
        {p.teams.length === 0 ? (
          <p className="text-sm text-muted">You are not in a team yet.</p>
        ) : (
          <ul className="divide-y divide-border">
            {p.teams.map((t) => (
              <li key={t.team_id} className="py-2">
                <div className="flex items-center justify-between gap-2">
                  <p className="truncate text-sm font-semibold">{t.name}</p>
                  <Pill>{ROLE_LABEL[t.team_role] ?? t.team_role}</Pill>
                </div>
                <p className="text-xs text-muted">
                  {t.member_code && <>Member code <span className="font-mono">{t.member_code}</span> · </>}
                  joined {dateOnly(t.joined_at)}
                </p>
                {t.git_identities.length > 0 && (
                  <p className="text-xs text-muted">Git names: {t.git_identities.join(", ")}</p>
                )}
              </li>
            ))}
          </ul>
        )}
      </Card>

      {student && (
        <>
          <Card title="Strengths" note="Worked out from your activity in the app. Each one says what it is based on.">
            <ItemList items={p.strengths} tone="good"
                      empty={p.has_activity ? "Nothing stands out yet. Keep committing, chatting and proposing decisions."
                        : "Not enough activity yet. Strengths appear as you commit code, chat and propose decisions."} />
          </Card>
          <Card title="Growth areas" note="Parts of the project you have done little of so far. A hint, not a judgement.">
            <ItemList items={p.growth} tone="grow"
                      empty={p.has_activity ? "Nothing to flag right now." : "Not enough activity yet to suggest anything."} />
          </Card>
          {p.work_mix && <WorkMix w={p.work_mix} />}
        </>
      )}

      <NotesCard notes={p.notes} onSaved={(notes) => setP({ ...p, notes })} />
    </>
  )
}
