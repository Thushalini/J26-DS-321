"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"

import { Button, Card, inputClass } from "@/components/ui"
import { api, shortDate, type Team } from "@/lib/api"
import { useSession } from "@/lib/session"
import { supervisorCopy as copy } from "@/lib/supervision"

type Review = Team & { expires_at: string }

export function PanelReview({ onError }: { onError: (message: string) => void }) {
  const { refreshTeams, selectTeam } = useSession()
  const router = useRouter()
  const [review, setReview] = useState<Review | null>(null)
  const [code, setCode] = useState("")
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api<Review | null>("/app/panel-review").then(setReview).catch((e) => onError(e.message))
  }, [onError])

  const act = async (fn: () => Promise<Review | null>) => {
    setBusy(true)
    try {
      const next = await fn()
      setReview(next)
      const teams = await refreshTeams()
      if (next) selectTeam(next.team_id)
      else {
        const fallback = teams.find((t) => t.my_role === "supervisor" && !t.archived) ?? teams[0]
        if (fallback) selectTeam(fallback.team_id)
      }
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const start = () => act(async () => {
    const next = await api<Review>("/app/panel-review", { method: "POST", json: { join_code: code } })
    setCode("")
    return next
  })

  const end = () => act(async () => {
    await api("/app/panel-review", { method: "DELETE" })
    return null
  })

  return (
    <Card title={copy.panelTitle} note={copy.panelNote}>
      <div className="flex flex-col gap-3">
        {review && (
          <div className="rounded-lg border border-research/40 bg-research/10 p-3">
            <p className="text-xs text-muted">{copy.panelCurrent}</p>
            <p className="text-sm font-semibold">{review.name}</p>
            <p className="text-xs text-muted">{copy.panelUntil} {shortDate(review.expires_at)}</p>
            <div className="mt-2 flex gap-2">
              <Button onClick={() => {
                selectTeam(review.team_id)
                router.push("/behavior")
              }}>{copy.panelOpen}</Button>
              <Button variant="secondary" onClick={end} disabled={busy}>{copy.panelEnd}</Button>
            </div>
          </div>
        )}
        <div className="flex gap-2">
          <input className={inputClass} placeholder={copy.panelPlaceholder} value={code}
                 onChange={(e) => setCode(e.target.value)} aria-label={copy.panelTitle} />
          <Button onClick={start} disabled={busy || !code.trim()}>{copy.panelStart}</Button>
        </div>
      </div>
    </Card>
  )
}
