"use client"
import type React from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useEffect, useState } from "react"

import { PasswordInput } from "@/components/password-input"
import { Button, Card, Field, Notice, inputClass } from "@/components/ui"
import { api, type User, type UserRole } from "@/lib/api"
import { useSession } from "@/lib/session"
import { site } from "@/lib/site"

const ROLES: { value: UserRole; label: string; hint: string }[] = [
  { value: "student", label: "Student (team member)", hint: "Create or join your team, chat, see your team's behaviour." },
  { value: "supervisor", label: "Supervisor", hint: "Supervise your team: ask the team lead for the join code, then join it from the Teams page." },
  { value: "examiner", label: "Examiner / researcher", hint: "Read-only access to the Phase 1 research dataset and all feedback. Needs the staff code from the project owner." },
]

export default function LoginPage() {
  const { ready, user, signIn } = useSession()
  const router = useRouter()
  const [mode, setMode] = useState<"login" | "register">("login")
  const [form, setForm] = useState({ email: "", password: "", confirm: "", name: "", role: "student" as UserRole, staff_code: "" })
  const [error, setError] = useState("")
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (ready && user) router.replace("/teams")
  }, [ready, user, router])

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: e.target.value })

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (mode === "register" && form.password !== form.confirm) {
      setError("The two passwords don't match.")
      return
    }
    setBusy(true)
    setError("")
    try {
      const { confirm: _confirm, ...account } = form
      const body = mode === "login" ? { email: form.email, password: form.password } : account
      const res = await api<{ token: string; user: User }>(`/app/auth/${mode}`, { method: "POST", json: body })
      signIn(res.token, res.user)
      router.replace("/teams")
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const needsStaffCode = form.role === "examiner"

  return (
    <main className="mx-auto flex min-h-dvh max-w-app flex-col justify-center gap-4 p-4">
      <div>
        <h1 className="text-2xl font-semibold">{site.name}</h1>
        <p className="mt-1 text-sm text-muted">
          Behavioural analytics for data-science teams: lifecycle stages, plan drift and member roles from your repo and team chat.
        </p>
      </div>

      <Card>
        <div className="mb-4 flex gap-2" role="tablist">
          {(["login", "register"] as const).map((m) => (
            <button
              key={m}
              type="button"
              role="tab"
              aria-selected={mode === m}
              onClick={() => {
                setMode(m)
                setError("")
              }}
              className={`flex-1 rounded-lg border px-3 py-2 text-sm font-medium ${mode === m ? "border-primary bg-primary text-primary-foreground" : "border-border"}`}
            >
              {m === "login" ? "Sign in" : "Create account"}
            </button>
          ))}
        </div>

        <form className="flex flex-col gap-3" onSubmit={submit}>
          {mode === "register" && (
            <>
              <Field label="Your name">
                <input className={inputClass} required value={form.name} onChange={set("name")} autoComplete="name" />
              </Field>
              <Field label="I am a" hint={ROLES.find((r) => r.value === form.role)?.hint}>
                <select className={inputClass} value={form.role} onChange={set("role")}>
                  {ROLES.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
                </select>
              </Field>
            </>
          )}
          <Field label="Email">
            <input className={inputClass} type="email" required value={form.email} onChange={set("email")} autoComplete="email" />
          </Field>
          <Field label="Password" hint={mode === "register" ? "At least 8 characters." : undefined}>
            <PasswordInput required minLength={mode === "register" ? 8 : 1} value={form.password}
                           onChange={set("password")} autoComplete={mode === "login" ? "current-password" : "new-password"} />
          </Field>
          {mode === "register" && (
            <Field label="Confirm password">
              <PasswordInput required minLength={8} value={form.confirm} onChange={set("confirm")} autoComplete="new-password" />
            </Field>
          )}
          {mode === "login" && (
            <Link href={`/forgot-password${form.email ? `?email=${encodeURIComponent(form.email)}` : ""}`}
                  className="-mt-1 self-end text-xs text-primary">
              Forgot password?
            </Link>
          )}
          {mode === "register" && needsStaffCode && (
            <Field label="Staff code" hint="Ask the project owner for this code.">
              <input className={inputClass} required value={form.staff_code} onChange={set("staff_code")} />
            </Field>
          )}
          {error && <Notice tone="error">{error}</Notice>}
          <Button type="submit" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}</Button>
        </form>
      </Card>

      <p className="text-center text-sm">
        <Link href="/guide" className="text-primary">How does Buddy.ds work for students, supervisors and examiners?</Link>
      </p>
    </main>
  )
}
