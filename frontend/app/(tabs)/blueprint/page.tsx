"use client"
// Blueprint tab. Top: the nine-field planning blueprint built by Component 1 (Buddy) from the chat.
// Below: the three-value plan that Component 2 measures drift against. It is read-only:
// Buddy fills it in once a value is known or confirmed.
import { useEffect, useState } from "react"

import { BlueprintPlan } from "@/components/blueprint-plan"
import { Card, Field, Notice, Pill } from "@/components/ui"
import { api, shortDate } from "@/lib/api"
import { useSession } from "@/lib/session"

interface Blueprint {
  source: "app" | "inferred" | "file" | "none"
  editable: boolean
  target: string | null
  algorithm: string | null
  metric: string | null
  notes: string | null
  updated_at: string | null
  updated_by: string | null
}

const FIELDS = [
  { key: "target", label: "Target variable" },
  { key: "algorithm", label: "Main algorithm" },
  { key: "metric", label: "Evaluation metric" },
] as const

const SOURCE_TEXT = {
  app: "Filled in from the team's chat",
  inferred: "Inferred from the first 40% of the project's activity (research teams have no written plan)",
  file: "Loaded from a blueprint file",
  none: "No blueprint yet",
}

const POLL_MS = 10000

export default function BlueprintPage() {
  const { team } = useSession()
  const [bp, setBp] = useState<Blueprint | null>(null)
  const [error, setError] = useState("")

  useEffect(() => {
    if (!team) return
    let alive = true
    setBp(null)
    setError("")
    const load = () =>
      api<Blueprint>(`/app/teams/${encodeURIComponent(team.team_id)}/blueprint`)
        .then((b) => { if (alive) { setBp(b); setError("") } })
        .catch((e) => { if (alive) setError(e.message) })
    load()
    const id = setInterval(load, POLL_MS)
    return () => { alive = false; clearInterval(id) }
  }, [team])

  if (!team) return null

  return (
    <div className="flex flex-col gap-4">
      {!team.is_research && <BlueprintPlan teamId={team.team_id} />}

      <Card
        title="Drift-check plan (target / algorithm / metric)"
        note="The three-value plan Component 2 checks the team's work against. Buddy fills it in from the chat once a value is known or confirmed. It cannot be edited by hand."
      >
        {bp && (
          <p className="flex flex-wrap items-center gap-2 text-xs text-muted">
            <Pill tone={bp.source === "inferred" ? "research" : "neutral"}>{SOURCE_TEXT[bp.source]}</Pill>
            {bp.updated_at && <span>Last changed {shortDate(bp.updated_at)}{bp.updated_by ? ` by ${bp.updated_by}` : ""}</span>}
          </p>
        )}
      </Card>

      {error && <Notice tone="error">{error}</Notice>}
      {!bp && !error && <p className="text-sm text-muted">Loading...</p>}

      {bp && (
        <Card>
          <div className="flex flex-col gap-3">
            {FIELDS.map((f) => (
              <Field key={f.key} label={f.label}>
                <span className="text-sm">{bp[f.key] ?? <span className="text-muted">Not set yet</span>}</span>
              </Field>
            ))}
            {bp.notes && <p className="text-sm text-muted">{bp.notes}</p>}
          </div>
        </Card>
      )}
    </div>
  )
}