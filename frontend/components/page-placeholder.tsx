import { getNavItem, type NavId } from "@/lib/navigation"

export function PagePlaceholder({ id }: { id: NavId }) {
  return (
    <section className="rounded-card border border-border bg-surface p-4">
      <h1 className="text-lg font-semibold">{getNavItem(id).label}</h1>
    </section>
  )
}
