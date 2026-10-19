"use client"

import { useEffect, useState } from "react"

import { Button, Card, Notice } from "@/components/ui"
import { api, type User } from "@/lib/api"
import { supervisorCopy as copy } from "@/lib/supervision"

export function SupervisorCode() {
  const [code, setCode] = useState<string | null>(null)
  const [error, setError] = useState("")
  const [done, setDone] = useState(false)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api<User>("/app/auth/me").then((u) => setCode(u.supervisor_code ?? null)).catch((e) => setError(e.message))
  }, [])

  const regenerate = async () => {
    if (!window.confirm(copy.codeRegenerateConfirm)) return
    setBusy(true)
    setError("")
    try {
      const r = await api<{ supervisor_code: string }>("/app/auth/supervisor-code", { method: "POST" })
      setCode(r.supervisor_code)
      setDone(true)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card title={copy.codeTitle} note={copy.codeNote}>
      <details>
        <summary className="cursor-pointer text-sm text-primary">{copy.codeShow}</summary>
        <div className="mt-3 flex flex-col gap-3">
          <div className="flex items-center justify-between rounded-lg border border-border bg-background p-3">
            <p className="font-mono text-lg font-semibold tracking-wider">{code ?? "…"}</p>
            <Button variant="secondary" disabled={!code} onClick={() => code && navigator.clipboard?.writeText(code)}>{copy.codeCopy}</Button>
          </div>
          <div><Button variant="ghost" className="px-0" onClick={regenerate} disabled={busy}>{copy.codeRegenerate}</Button></div>
          {done && <Notice tone="success">{copy.codeRegenerated}</Notice>}
          {error && <Notice tone="error">{error}</Notice>}
        </div>
      </details>
    </Card>
  )
}
