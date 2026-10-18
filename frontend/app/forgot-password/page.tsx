"use client"
// Forgot password: email a 6-digit code, then set a new password with it (signs you in).
import type React from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useEffect, useState } from "react"

import { PasswordInput } from "@/components/password-input"
import { Button, Card, Field, Notice, inputClass } from "@/components/ui"
import { api, type User } from "@/lib/api"
import { useSession } from "@/lib/session"
import { site } from "@/lib/site"

export default function ForgotPasswordPage() {
  const { signIn } = useSession()
  const router = useRouter()
  const [step, setStep] = useState<"email" | "code">("email")
  const [form, setForm] = useState({ email: "", code: "", password: "", confirm: "" })
  const [error, setError] = useState("")
  const [info, setInfo] = useState("")
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const email = new URLSearchParams(window.location.search).get("email")
    if (email) setForm((f) => ({ ...f, email }))
  }, [])

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: e.target.value })

  const sendCode = async (e?: React.FormEvent) => {
    e?.preventDefault()
    setBusy(true)
    setError("")
    try {
      await api("/app/auth/forgot", { method: "POST", json: { email: form.email } })
      setInfo(`If an account exists for ${form.email}, a 6-digit code is on its way. It works for 15 minutes.`)
      setStep("code")
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const reset = async (e: React.FormEvent) => {
    e.preventDefault()
    if (form.password !== form.confirm) {
      setError("The two passwords don't match.")
      return
    }
    setBusy(true)
    setError("")
    try {
      const res = await api<{ token: string; user: User }>("/app/auth/reset", {
        method: "POST",
        json: { email: form.email, code: form.code, new_password: form.password },
      })
      signIn(res.token, res.user)
      router.replace("/teams")
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="mx-auto flex min-h-dvh max-w-app flex-col justify-center gap-4 p-4">
      <div>
        <h1 className="text-2xl font-semibold">{site.name}</h1>
        <p className="mt-1 text-sm text-muted">Reset your password</p>
      </div>

      <Card>
        {step === "email" ? (
          <form className="flex flex-col gap-3" onSubmit={sendCode}>
            <Field label="Email" hint="We'll send a reset code to this address.">
              <input className={inputClass} type="email" required value={form.email} onChange={set("email")} autoComplete="email" />
            </Field>
            {error && <Notice tone="error">{error}</Notice>}
            <Button type="submit" disabled={busy}>{busy ? "Please wait…" : "Send reset code"}</Button>
          </form>
        ) : (
          <form className="flex flex-col gap-3" onSubmit={reset}>
            {info && <Notice tone="success">{info}</Notice>}
            <Field label="Reset code">
              <input className={inputClass} required inputMode="numeric" pattern="[0-9]{6}" maxLength={6} value={form.code}
                     onChange={set("code")} autoComplete="one-time-code" />
            </Field>
            <Field label="New password" hint="At least 8 characters.">
              <PasswordInput required minLength={8} value={form.password} onChange={set("password")} autoComplete="new-password" />
            </Field>
            <Field label="Confirm new password">
              <PasswordInput required minLength={8} value={form.confirm} onChange={set("confirm")} autoComplete="new-password" />
            </Field>
            {error && <Notice tone="error">{error}</Notice>}
            <Button type="submit" disabled={busy}>{busy ? "Please wait…" : "Set new password"}</Button>
            <Button variant="ghost" disabled={busy} onClick={() => sendCode()}>Send a new code</Button>
          </form>
        )}
      </Card>

      <p className="text-center text-sm">
        <Link href="/login" className="text-primary">Back to sign in</Link>
      </p>
    </main>
  )
}
