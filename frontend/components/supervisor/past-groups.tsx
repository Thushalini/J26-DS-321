"use client"

import { useEffect, useState } from "react"

import { Button, Card, Notice } from "@/components/ui"
import { api } from "@/lib/api"
import { useSession } from "@/lib/session"
import { supervisorCopy as copy } from "@/lib/supervision"

export function PastGroups() {
  const { teams, refreshTeams } = useSession()
  const [error, setError] = useState("")
  const [busy, setBusy] = useState(false)
  const past = teams.filter((t) => t.my_role === "supervisor" && t.archived)

  useEffect(() => {
    refreshTeams().catch((e) => setError(e.message))
  }, [refreshTeams])

  const restore = async (teamId: string) => {
    setBusy(true)
    try {
      await api(`/app/teams/${encodeURIComponent(teamId)}/supervision`, { method: "PUT", json: { archived: false } })
      await refreshTeams()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  if (!past.length && !error) return null

  return (
    <Card title={copy.pastTitle} note={copy.pastNote}>
      {error && <Notice tone="error">{error}</Notice>}
      <ul className="flex flex-col divide-y divide-border">
        {past.map((t) => (
          <li key={t.team_id} className="flex items-center justify-between gap-2 py-2 text-sm">
            <span className="min-w-0 truncate">{t.name}</span>
            <Button variant="secondary" onClick={() => restore(t.team_id)} disabled={busy}>{copy.restore}</Button>
          </li>
        ))}
      </ul>
    </Card>
  )
}
