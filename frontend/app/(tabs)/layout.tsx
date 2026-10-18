import type React from "react"
import { AppShell } from "@/components/app-shell"

export default function TabsLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return <AppShell>{children}</AppShell>
}
