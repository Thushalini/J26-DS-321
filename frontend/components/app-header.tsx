"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"

import { Pill } from "@/components/ui"
import { ROLE_LABEL } from "@/lib/api"
import { isSupervisorLayout } from "@/lib/navigation"
import { useSession } from "@/lib/session"
import { site } from "@/lib/site"
import { supervisorHeaderCopy, supervisorLayout } from "@/lib/supervision"

export function AppHeader() {
  const { user, teams, team, selectTeam, signOut } = useSession()
  const router = useRouter()
  const staff = user?.role === "supervisor" || user?.role === "examiner"

  return (
    <header className="sticky top-0 z-10 border-b border-border bg-surface">
      <div className="flex items-center justify-between gap-2 px-4 py-2">
        <Link href="/teams" className="text-sm font-semibold">{site.name}</Link>
        <div className="flex items-center gap-3 text-xs">
          <Link href="/guide" className="text-primary">Guide</Link>
          {staff && <Link href="/feedback" className="text-primary">Feedback</Link>}
          <Link href="/account" className="text-primary">Change password</Link>
          <button
            type="button"
            className="text-muted hover:text-foreground"
            onClick={async () => {
              await signOut()
              router.replace("/login")
            }}
          >
            Sign out
          </button>
        </div>
      </div>
      {isSupervisorLayout(user?.role) ? (
        <div className="flex items-center justify-between gap-2 border-t border-border px-4 py-2">
          <div className="min-w-0">
            <p className="text-xs text-muted">{supervisorHeaderCopy.currentGroup}</p>
            <p className="truncate text-sm font-semibold">{team?.name ?? supervisorHeaderCopy.noGroup}</p>
          </div>
          <div className="flex shrink-0 items-center gap-3">
            {!supervisorLayout.researchInGroupList && teams.some((t) => t.is_research) && (
              <select
                aria-label={supervisorHeaderCopy.researchLabel}
                className="max-w-36 truncate rounded-lg border border-border bg-surface px-2 py-1 text-xs"
                value={team?.is_research ? team.team_id : ""}
                onChange={(e) => {
                  if (!e.target.value) return
                  selectTeam(e.target.value)
                  router.push(supervisorLayout.openGroupAt)
                }}
              >
                <option value="">{supervisorHeaderCopy.researchPlaceholder}</option>
                {teams.filter((t) => t.is_research).map((t) => <option key={t.team_id} value={t.team_id}>{t.name}</option>)}
              </select>
            )}
            <Link href={supervisorLayout.landing} className="text-xs font-medium text-primary">
              {supervisorHeaderCopy.myGroups}
            </Link>
          </div>
        </div>
      ) : (
      <div className="flex items-center gap-2 border-t border-border px-4 py-2">
        <label className="sr-only" htmlFor="team-picker">Team</label>
        <select
          id="team-picker"
          className="min-w-0 flex-1 truncate rounded-lg border border-border bg-surface px-2 py-1.5 text-sm"
          value={team?.team_id ?? ""}
          onChange={(e) => (e.target.value === "__manage" ? router.push("/teams") : selectTeam(e.target.value))}
        >
          {!team && <option value="">No team yet</option>}
          {teams.some((t) => !t.is_research && t.my_role !== "panel" && !t.archived) && (
            <optgroup label={user?.role === "examiner" ? "Current teams (anonymised)" : "My teams"}>
              {teams.filter((t) => !t.is_research && t.my_role !== "panel" && !t.archived)
                .map((t) => <option key={t.team_id} value={t.team_id}>{t.name}</option>)}
            </optgroup>
          )}
          {teams.some((t) => t.my_role === "panel") && (
            <optgroup label="Panel review">
              {teams.filter((t) => t.my_role === "panel").map((t) => <option key={t.team_id} value={t.team_id}>{t.name}</option>)}
            </optgroup>
          )}
          {teams.some((t) => t.is_research) && (
            <optgroup label="Research dataset (Phase 1)">
              {teams.filter((t) => t.is_research).map((t) => <option key={t.team_id} value={t.team_id}>{t.name}</option>)}
            </optgroup>
          )}
          <option value="__manage">+ Create / join / manage teams…</option>
        </select>
        {user && <Pill tone="primary">{user.name.split(" ")[0]} · {ROLE_LABEL[team?.my_role ?? user.role]}</Pill>}
      </div>
      )}
      {team?.anonymised && (
        <p className="border-t border-research/40 bg-research/15 px-4 py-1.5 text-xs">
          Current team · read-only, anonymised: member codes only, no names, repo link or chat
        </p>
      )}
      {team?.is_research && (
        <p className="border-t border-research/40 bg-research/15 px-4 py-1.5 text-xs">
          {team.label} · read-only, member codes are anonymised
        </p>
      )}
    </header>
  )
}
