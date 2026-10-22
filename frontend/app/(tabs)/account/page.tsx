"use client"
// My profile (private): about me, teams, strengths and growth areas, notes (components/my-profile.tsx),
// then change password (other devices get signed out, this one stays signed in).
import type React from "react"
import { useState } from "react"

import { MyProfile } from "@/components/my-profile"
import { PasswordInput } from "@/components/password-input"
import { Button, Card, Field, Notice } from "@/components/ui"
import { api } from "@/lib/api"

const EMPTY = { current: "", password: "", confirm: "" }

export default function AccountPage() {
  const [form, setForm] = useState(EMPTY)
  const [error, setError] = useState("")
  const [done, setDone] = useState(false)
  const [busy, setBusy] = useState(false)

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: e.target.value })

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setDone(false)
    if (form.password !== form.confirm) {
      setError("The two new passwords don't match.")
      return
    }
    setBusy(true)
    setError("")
    try {
      await api("/app/auth/change-password", {
        method: "POST",
        json: { current_password: form.current, new_password: form.password },
      })
      setForm(EMPTY)
      setDone(true)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <MyProfile />

      <Card title="Change password" note="You stay signed in here; any other devices are signed out.">
        <form className="flex flex-col gap-3" onSubmit={submit}>
          <Field label="Current password">
            <PasswordInput required value={form.current} onChange={set("current")} autoComplete="current-password" />
          </Field>
          <Field label="New password" hint="At least 8 characters.">
            <PasswordInput required minLength={8} value={form.password} onChange={set("password")} autoComplete="new-password" />
          </Field>
          <Field label="Confirm new password">
            <PasswordInput required minLength={8} value={form.confirm} onChange={set("confirm")} autoComplete="new-password" />
          </Field>
          {error && <Notice tone="error">{error}</Notice>}
          {done && <Notice tone="success">Your password has been changed.</Notice>}
          <Button type="submit" disabled={busy}>{busy ? "Please wait…" : "Change password"}</Button>
        </form>
      </Card>
    </div>
  )
}
