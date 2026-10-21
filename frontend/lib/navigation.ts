import { supervisorLayout } from "@/lib/supervision"

export const navItems = [
  { id: "chat", label: "Chat", href: "/chat" },
  { id: "pulse", label: "Pulse", href: "/pulse" },
  { id: "blueprint", label: "Blueprint", href: "/blueprint" },
  { id: "behavior", label: "Behavior", href: "/behavior" },
  { id: "alerts", label: "Alerts", href: "/alerts" },
  { id: "predict", label: "Predict", href: "/predict" },
  { id: "summary", label: "Summary", href: "/summary" },
  { id: "home", label: "Home", href: "/teams" },
] as const

export type NavItem = (typeof navItems)[number]
export type NavId = NavItem["id"]

const studentTabs: NavId[] = ["chat", "pulse", "blueprint", "behavior", "alerts", "predict"]

export const defaultNavItem: NavItem = navItems[0]

export function getNavItem(id: NavId): NavItem {
  return navItems.find((item) => item.id === id) ?? defaultNavItem
}

export function isSupervisorLayout(role: string | undefined): boolean {
  return role === "supervisor" && supervisorLayout.groupListAsPicker
}

export function navItemsFor(role: string | undefined): NavItem[] {
  const ids: readonly NavId[] = isSupervisorLayout(role) ? supervisorLayout.tabs : studentTabs
  return ids.map(getNavItem)
}
