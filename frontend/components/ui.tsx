// Small shared building blocks so every screen looks the same.
import type React from "react"

export function Card({ title, note, children, className = "" }: {
  title?: string
  note?: React.ReactNode
  children?: React.ReactNode
  className?: string
}) {
  return (
    <section className={`rounded-card border border-border bg-surface p-4 ${className}`}>
      {title && <h2 className="text-base font-semibold">{title}</h2>}
      {note && <p className="mt-1 text-sm text-muted">{note}</p>}
      {children && <div className={title || note ? "mt-3" : ""}>{children}</div>}
    </section>
  )
}

export function Button({ variant = "primary", className = "", ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost"
}) {
  const look = {
    primary: "bg-primary text-primary-foreground border-primary",
    secondary: "bg-surface text-foreground border-border",
    ghost: "bg-transparent text-primary border-transparent",
  }[variant]
  return (
    <button
      type="button"
      {...props}
      className={`rounded-lg border px-3 py-2 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary ${look} ${className}`}
    />
  )
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="font-medium">{label}</span>
      {children}
      {hint && <span className="text-xs text-muted">{hint}</span>}
    </label>
  )
}

export const inputClass =
  "w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-foreground focus:outline-2 focus:outline-primary"

export function Notice({ tone = "info", children }: { tone?: "info" | "error" | "success" | "research"; children: React.ReactNode }) {
  const look = {
    info: "border-border bg-background text-foreground",
    error: "border-danger/40 bg-danger/10 text-danger",
    success: "border-success/40 bg-success/10 text-success",
    research: "border-research/40 bg-research/10 text-foreground",
  }[tone]
  return (
    <p role={tone === "error" ? "alert" : undefined} className={`rounded-lg border px-3 py-2 text-sm ${look}`}>
      {children}
    </p>
  )
}

export function Pill({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "research" | "primary" }) {
  const look = {
    neutral: "border-border bg-background text-muted",
    research: "border-research/40 bg-research/15 text-foreground",
    primary: "border-primary/30 bg-primary/10 text-primary",
  }[tone]
  return <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${look}`}>{children}</span>
}
