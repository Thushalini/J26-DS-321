import { getNavItem, type NavId } from "@/lib/navigation"

export function PagePlaceholder({ id, note }: { id: NavId; note?: string }) {
  return (
    <section className="rounded-card border border-border bg-surface p-4">
      <h1 className="text-lg font-semibold">{getNavItem(id).label}</h1>
      {note && <p className="mt-1 text-sm text-muted">{note}</p>}
    </section>
  )
}
