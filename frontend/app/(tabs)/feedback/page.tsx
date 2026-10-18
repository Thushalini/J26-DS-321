"use client"
// Feedback collected from every screen (supervisors: their teams, students anonymous; examiners: all).
import { useEffect, useState } from "react"

import { Button, Card, Notice, Pill } from "@/components/ui"
import { api, API, getToken, ROLE_LABEL, shortDate } from "@/lib/api"
import { useSession } from "@/lib/session"

interface Item {
  ts: string
  name: string
  role: string
  team_id: string | null
  team_name: string | null
  page: string | null
  rating: number | null
  comment: string | null
}

interface Stats {
  count: number
  average_rating: number | null
}

interface FeedbackList extends Stats {
  by_role: Record<string, Stats>
  items: Item[]
}

export default function FeedbackPage() {
  const { user } = useSession()
  const [data, setData] = useState<FeedbackList | null>(null)
  const [error, setError] = useState("")

  useEffect(() => {
    api<FeedbackList>("/app/feedback").then(setData).catch((e) => setError(e.message))
  }, [])

  const download = async () => {
    const res = await fetch(`${API}/app/feedback.csv`, { headers: { Authorization: `Bearer ${getToken()}` } })
    const url = URL.createObjectURL(await res.blob())
    const a = document.createElement("a")
    a.href = url
    a.download = "buddy_feedback.csv"
    a.click()
    URL.revokeObjectURL(url)
  }

  if (user?.role === "student") return <Notice>Only supervisors and examiners can read feedback.</Notice>

  return (
    <div className="flex flex-col gap-4">
      <Card title="Feedback"
            note={user?.role === "examiner" ? "All feedback from every role." : "Feedback from the teams you supervise (students stay anonymous), plus your own."}>
        {data && (
          <div className="flex flex-col gap-3">
            <dl className="grid grid-cols-2 gap-2">
              <div className="rounded-lg border border-border p-3"><dt className="text-xs text-muted">Responses</dt><dd className="text-xl font-semibold">{data.count}</dd></div>
              <div className="rounded-lg border border-border p-3"><dt className="text-xs text-muted">Average rating</dt><dd className="text-xl font-semibold">{data.average_rating ?? "–"} <span className="text-sm text-muted">/ 5</span></dd></div>
            </dl>
            <p className="flex flex-wrap gap-2 text-xs">
              {Object.entries(data.by_role).map(([role, s]) => (
                <Pill key={role}>{ROLE_LABEL[role] ?? role}: {s.count} · avg {s.average_rating ?? "–"}</Pill>
              ))}
            </p>
            <div><Button variant="secondary" onClick={download} disabled={!data.count}>Download CSV</Button></div>
          </div>
        )}
      </Card>
      {error && <Notice tone="error">{error}</Notice>}
      <ul className="flex flex-col gap-2">
        {data?.items.map((f, i) => (
          <li key={i} className="rounded-card border border-border bg-surface p-3">
            <div className="flex items-center justify-between gap-2 text-xs text-muted">
              <span>{f.name} · {ROLE_LABEL[f.role] ?? f.role}</span>
              <span>{shortDate(f.ts)}</span>
            </div>
            <p className="mt-1 text-xs text-muted">{f.page}{f.team_name ? ` · ${f.team_name}` : ""}</p>
            {f.rating && <p className="mt-1 text-sm font-semibold">{"★".repeat(f.rating)}{"☆".repeat(5 - f.rating)} <span className="sr-only">{f.rating} of 5</span></p>}
            {f.comment && <p className="mt-1 whitespace-pre-wrap text-sm">{f.comment}</p>}
          </li>
        ))}
      </ul>
    </div>
  )
}
