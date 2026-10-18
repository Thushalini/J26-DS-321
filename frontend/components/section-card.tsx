import type React from "react"

type SectionCardProps = {
  title: string
  description?: string
  aside?: React.ReactNode
  children: React.ReactNode
}

export function SectionCard({ title, description, aside, children }: SectionCardProps) {
  return (
    <section className="rounded-card border border-border bg-surface p-4">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold">{title}</h2>
          {description && <p className="mt-0.5 text-xs text-muted">{description}</p>}
        </div>
        {aside}
      </header>
      <div className="mt-4">{children}</div>
    </section>
  )
}
