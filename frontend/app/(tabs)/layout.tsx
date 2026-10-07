import type React from "react"
import { AppHeader } from "@/components/app-header"
import { BottomNav } from "@/components/bottom-nav"

export default function TabsLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <div className="mx-auto flex min-h-dvh max-w-app flex-col">
      <AppHeader />
      <main className="flex-1 p-4">{children}</main>
      <BottomNav />
    </div>
  )
}
