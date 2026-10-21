"use client"
// Signed-in frame: header (team picker), page, feedback button, bottom tabs.
// Not signed in -> /login. Signed in without any team -> /teams (create or join one).
import type React from "react"
import { usePathname, useRouter } from "next/navigation"
import { useEffect } from "react"

import { AppHeader } from "@/components/app-header"
import { BottomNav } from "@/components/bottom-nav"
import { FeedbackButton } from "@/components/feedback-button"
import { isSupervisorLayout } from "@/lib/navigation"
import { useSession } from "@/lib/session"
import { supervisorLayout } from "@/lib/supervision"

const NO_TEAM_OK = ["/teams", "/feedback", "/account"]

export function AppShell({ children }: { children: React.ReactNode }) {
  const { ready, user, team } = useSession()
  const router = useRouter()
  const pathname = usePathname()

  useEffect(() => {
    if (!ready) return
    if (!user) router.replace("/login")
    else if (isSupervisorLayout(user.role) && !supervisorLayout.allowedPaths.some((p) => pathname.startsWith(p)))
      router.replace(supervisorLayout.landing)
    else if (!team && !NO_TEAM_OK.some((p) => pathname.startsWith(p))) router.replace("/teams")
  }, [ready, user, team, pathname, router])

  if (!ready || !user) {
    return <p className="p-8 text-center text-sm text-muted">Loading…</p>
  }

  return (
    <div className="mx-auto flex min-h-dvh max-w-app flex-col">
      <AppHeader />
      <main className="flex-1 p-4 pb-20">{children}</main>
      <FeedbackButton />
      <BottomNav />
    </div>
  )
}
