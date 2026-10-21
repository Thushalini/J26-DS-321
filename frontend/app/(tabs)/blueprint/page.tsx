"use client"
// Blueprint tab. Top: the nine-field planning blueprint built by Component 1 (Buddy) from the chat.
// Below: the three-value plan that Component 2 measures drift against (entered by hand).
import { useEffect, useState } from "react"

import { BlueprintPlan } from "@/components/blueprint-plan"
import { Button, Card, Field, Notice, Pill, inputClass } from "@/components/ui"
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
  { key: "target", label: "Target variable", hint: "What you predict, e.g. house price, churn, diabetes" },
  { key: "algorithm", label: "Main algorithm", hint: "e.g. Random Forest, XGBoost, Logistic Regression" },
  { key: "metric", label: "Evaluation metric", hint: "e.g. RMSE, F1, accuracy, AUC" },
] as const

const SOURCE_TEXT = {
  app: "Entered by the team",
  inferred: "Inferred from the first 40% of the project's activity (research teams have no written plan)",
  file: "Loaded from a blueprint file",
  none: "No blueprint yet",
}

export default function BlueprintPage() {
  const { team } = useSession()
  const [bp, setBp] = useState<Blueprint | null>(null)
  const [form, setForm] = useState({ target: "", algorithm: "", metric: "", notes: "" })
  const [error, setError] = useState("")
  const [msg, setMsg] = useState("")
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!team) return
    setBp(null)
    setMsg("")
    api<Blueprint>(`/app/teams/${encodeURIComponent(team.team_id)}/blueprint`)
      .then((b) => {
        setBp(b)
        setForm({ target: b.target ?? "", algorithm: b.algorithm ?? "", metric: b.metric ?? "", notes: b.notes ?? "" })
      })
      .catch((e) => setError(e.message))
  }, [team])

  const save = async () => {
    if (!team) return
    setSaving(true)
    setError("")
    setMsg("")
    try {
      const b = await api<Blueprint>(`/app/teams/${encodeURIComponent(team.team_id)}/blueprint`, { method: "PUT", json: form })
      setBp(b)
      setMsg("Saved. Drift on the Behavior tab is now measured against this plan.")
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  if (!team) return null

  return (
    <div className="flex flex-col gap-4">
      {!team.is_research && <BlueprintPlan teamId={team.team_id} />}

      <Card
        title="Drift-check plan (target / algorithm / metric)"
        note="The three-value plan Component 2 checks the team's work against. Buddy copies a value here once it is known or confirmed; a value typed by hand is never overwritten."
      >
        {bp && (
          <p className="flex flex-wrap items-center gap-2 text-xs text-muted">
            <Pill tone={bp.source === "inferred" ? "research" : "neutral"}>{SOURCE_TEXT[bp.source]}</Pill>
            {bp.updated_at && <span>Last changed {shortDate(bp.updated_at)}{bp.updated_by ? ` by ${bp.updated_by}` : ""}</span>}
          </p>
        )}
      </Card>

      {error && <Notice tone="error">{error}</Notice>}
      {msg && <Notice tone="success">{msg}</Notice>}
      {!bp && !error && <p className="text-sm text-muted">Loading...</p>}

      {bp && (
        <Card>
          <div className="flex flex-col gap-3">
            {FIELDS.map((f) => (
              <Field key={f.key} label={f.label} hint={bp.editable ? f.hint : undefined}>
                {bp.editable ? (
                  <input className={inputClass} value={form[f.key]} onChange={(e) => setForm({ ...form, [f.key]: e.target.value })} />
                ) : (
                  <span className="text-sm">{bp[f.key] ?? <span className="text-muted">Not set</span>}</span>
                )}
              </Field>
            ))}
            {bp.editable ? (
              <>
                <Field label="Notes (optional)">
                  <textarea className={`${inputClass} h-20`} value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} />
                </Field>
                <div><Button onClick={save} disabled={saving}>{saving ? "Saving and re-checking drift..." : "Save blueprint"}</Button></div>
              </>
            ) : (
              bp.notes && <p className="text-sm text-muted">{bp.notes}</p>
            )}
            {!bp.editable && !team.is_research && <p className="text-xs text-muted">Only team members can change the plan.</p>}
          </div>
        </Card>
      )}
    </div>
  )
}