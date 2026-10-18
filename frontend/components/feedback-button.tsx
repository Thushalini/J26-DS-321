"use client"
// "Give feedback" on every screen: a 1-5 rating and a comment, saved with the page and team.
import { usePathname } from "next/navigation"
import { useState } from "react"

import { Button, Notice, inputClass } from "@/components/ui"
import { api } from "@/lib/api"
import { useSession } from "@/lib/session"

export function FeedbackButton() {
  const { team } = useSession()
  const page = usePathname()
  const [open, setOpen] = useState(false)
  const [rating, setRating] = useState<number | null>(null)
  const [comment, setComment] = useState("")
  const [state, setState] = useState<"idle" | "sending" | "sent">("idle")
  const [error, setError] = useState("")

  const send = async () => {
    setState("sending")
    setError("")
    try {
      await api("/app/feedback", { method: "POST", json: { team_id: team?.team_id ?? null, page, rating, comment } })
      setState("sent")
      setRating(null)
      setComment("")
      setTimeout(() => {
        setOpen(false)
        setState("idle")
      }, 1500)
    } catch (e) {
      setError((e as Error).message)
      setState("idle")
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="fixed bottom-16 right-4 z-20 rounded-full bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground shadow-md"
      >
        Give feedback
      </button>
      {open && (
        <div className="fixed inset-0 z-30 flex items-end justify-center bg-black/30 sm:items-center" role="dialog" aria-modal="true" aria-labelledby="fb-title">
          <div className="w-full max-w-app rounded-t-card border border-border bg-surface p-4 sm:rounded-card">
            <h2 id="fb-title" className="text-base font-semibold">How useful is this screen?</h2>
            <p className="mt-1 text-xs text-muted">Screen: {page}{team ? ` · team: ${team.name}` : ""}</p>
            <div className="mt-3 flex gap-2" role="group" aria-label="Rating from 1 to 5">
              {[1, 2, 3, 4, 5].map((n) => (
                <button
                  key={n}
                  type="button"
                  aria-pressed={rating === n}
                  onClick={() => setRating(n)}
                  className={`h-10 flex-1 rounded-lg border text-sm font-semibold ${rating === n ? "border-primary bg-primary text-primary-foreground" : "border-border bg-surface"}`}
                >
                  {n}
                </button>
              ))}
            </div>
            <div className="mt-1 flex justify-between text-xs text-muted"><span>Not useful</span><span>Very useful</span></div>
            <textarea
              className={`${inputClass} mt-3 h-24`}
              placeholder="What was clear, wrong or missing? Did the roles / drift match what really happened?"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
            />
            {error && <div className="mt-2"><Notice tone="error">{error}</Notice></div>}
            {state === "sent" && <div className="mt-2"><Notice tone="success">Thank you, your feedback was saved.</Notice></div>}
            <div className="mt-3 flex justify-end gap-2">
              <Button variant="secondary" onClick={() => setOpen(false)}>Cancel</Button>
              <Button onClick={send} disabled={state !== "idle" || (!rating && !comment.trim())}>
                {state === "sending" ? "Sending…" : "Send"}
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
