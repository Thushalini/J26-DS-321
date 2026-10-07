import { site } from "@/lib/site"

export function AppHeader() {
  return (
    <header className="border-b border-border bg-surface px-4 py-3">
      <span className="text-sm font-semibold">{site.name}</span>
    </header>
  )
}
