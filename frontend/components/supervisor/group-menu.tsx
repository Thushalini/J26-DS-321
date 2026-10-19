"use client"

import { useEffect, useRef, useState } from "react"

import { Button, inputClass } from "@/components/ui"
import { api, type Team } from "@/lib/api"
import { supervisionStatuses, supervisorCopy as copy, type SupervisionStatus } from "@/lib/supervision"

export function GroupMenu({ team, onChanged, onError }: {
  team: Team
  onChanged: () => Promise<unknown>
  onError: (message: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const [name, setName] = useState(team.name)
  const [busy, setBusy] = useState(false)
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !box.current?.contains(e.target as Node)) {
        setOpen(false)
        setRenaming(false)
      }
    }
    document.addEventListener("mousedown", close)
    document.addEventListener("keydown", close)
    return () => {
      document.removeEventListener("mousedown", close)
      document.removeEventListener("keydown", close)
    }
  }, [open])

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try {
      await fn()
      await onChanged()
      setOpen(false)
      setRenaming(false)
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const supervision = (json: { status?: SupervisionStatus | null; archived?: boolean }) =>
    api(`/app/teams/${encodeURIComponent(team.team_id)}/supervision`, { method: "PUT", json })

  return (
    <div ref={box} className="relative">
      <button type="button" aria-label={copy.menuLabel} aria-haspopup="menu" aria-expanded={open}
              onClick={() => setOpen(!open)}
              className="rounded-lg px-2 py-1 text-lg leading-none text-muted hover:bg-background hover:text-foreground">
        ⋯
      </button>
      {open && (
        <div role="menu" className="absolute right-0 top-full z-20 mt-1 flex w-56 flex-col gap-1 rounded-lg border border-border bg-surface p-2 shadow-lg">
          {renaming ? (
            <div className="flex flex-col gap-2">
              <input className={inputClass} value={name} onChange={(e) => setName(e.target.value)} aria-label={copy.rename} autoFocus />
              <Button disabled={busy || name.trim().length < 2}
                      onClick={() => act(() => api(`/app/teams/${encodeURIComponent(team.team_id)}`, { method: "PUT", json: { name } }))}>
                {copy.save}
              </Button>
            </div>
          ) : (
            <>
              <button role="menuitem" type="button" className="rounded px-2 py-1.5 text-left text-sm hover:bg-background" onClick={() => setRenaming(true)}>
                {copy.rename}
              </button>
              <p className="px-2 pt-1 text-xs text-muted">{copy.setStatus}</p>
              {supervisionStatuses.map((s) => (
                <button key={s.id} role="menuitem" type="button" disabled={busy}
                        className={`flex items-center gap-2 rounded px-2 py-1.5 text-left text-sm hover:bg-background ${team.supervision_status === s.id ? "font-semibold" : ""}`}
                        onClick={() => act(() => supervision({ status: s.id }))}>
                  <span className={`size-2.5 rounded-full ${s.dot}`} aria-hidden="true" />
                  {s.label}
                </button>
              ))}
              {team.supervision_status && (
                <button role="menuitem" type="button" disabled={busy} className="rounded px-2 py-1.5 text-left text-sm text-muted hover:bg-background"
                        onClick={() => act(() => supervision({ status: null }))}>
                  {copy.clearStatus}
                </button>
              )}
              <hr className="my-1 border-border" />
              <button role="menuitem" type="button" disabled={busy} className="rounded px-2 py-1.5 text-left text-sm text-danger hover:bg-background"
                      onClick={() => window.confirm(copy.removeConfirm) && act(() => supervision({ archived: true }))}>
                {copy.remove}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  )
}
