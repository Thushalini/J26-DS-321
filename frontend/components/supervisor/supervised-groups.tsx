"use client"

import { Card } from "@/components/ui"
import { GroupMenu } from "@/components/supervisor/group-menu"
import { StatusDot } from "@/components/supervisor/status-dot"
import { shortDate, type Team } from "@/lib/api"
import { supervisorCopy as copy } from "@/lib/supervision"

export function SupervisedGroups({ teams, openId, onOpen, onChanged, onError }: {
  teams: Team[]
  openId: string | undefined
  onOpen: (teamId: string) => void
  onChanged: () => Promise<unknown>
  onError: (message: string) => void
}) {
  return (
    <Card title={copy.groupsTitle} note={teams.length ? copy.groupsNote : copy.noGroups}>
      {teams.length > 0 && (
        <ul className="flex flex-col gap-2">
          {teams.map((t) => (
            <li key={t.team_id}
                className={`flex items-center gap-2 rounded-lg border p-2 ${openId === t.team_id ? "border-primary bg-primary/5" : "border-border bg-surface"}`}>
              <button type="button" onClick={() => onOpen(t.team_id)} className="min-w-0 flex-1 p-1 text-left">
                <span className="block truncate text-sm font-semibold">{t.name}</span>
                <span className="block text-xs text-muted">
                  {t.members ?? 0} {t.members === 1 ? "member" : "members"} · {t.activities ?? 0} activities · synced {shortDate(t.last_synced)}
                </span>
              </button>
              <StatusDot status={t.supervision_status} />
              <GroupMenu team={t} onChanged={onChanged} onError={onError} />
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}
