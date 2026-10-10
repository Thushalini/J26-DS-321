export const navItems = [
  { id: "chat", label: "Chat", href: "/chat" },
  { id: "pulse", label: "Pulse", href: "/pulse" },
  { id: "blueprint", label: "Blueprint", href: "/blueprint" },
  { id: "behavior", label: "Behavior", href: "/behavior" },
  { id: "alerts", label: "Alerts", href: "/alerts" },
  { id: "predict", label: "Predict", href: "/predict" },
] as const

export type NavItem = (typeof navItems)[number]
export type NavId = NavItem["id"]

export const defaultNavItem: NavItem = navItems[0]

export function getNavItem(id: NavId): NavItem {
  return navItems.find((item) => item.id === id) ?? defaultNavItem
}
