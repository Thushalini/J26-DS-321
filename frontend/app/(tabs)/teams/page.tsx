"use client"
// Teams: create / join (students, supervisors), overview of every visible team (supervisor view),
// and the open team's members, join code, repo link and git-identity linking.
import { useCallback, useEffect, useState } from "react"
import { useRouter } from "next/navigation"

import { PanelReview } from "@/components/supervisor/panel-review"
import { PastGroups } from "@/components/supervisor/past-groups"
import { SupervisorCode } from "@/components/supervisor/supervisor-code"
import { SupervisedGroups } from "@/components/supervisor/supervised-groups"
import { isSupervisorLayout } from "@/lib/navigation"
import { supervisorLayout } from "@/lib/supervision"
import { Button, Card, Field, Notice, Pill, inputClass } from "@/components/ui"
import { api, ROLE_LABEL, shortDate, type Team, type TeamDetail } from "@/lib/api"
import { useSession } from "@/lib/session"

const SEVERITY_TEXT: Record<string, string> = {
  none: "On plan", minor: "Minor drift", moderate: "Moderate drift", significant: "Significant drift",
  unknown: "Not enough evidence", no_plan: "No plan yet",
}

function TeamRow({ t, open, onOpen }: { t: Team; open: boolean; onOpen: () => void }) {
  return (
    <li>
      <button type="button" onClick={onOpen}
              className={`w-full rounded-lg border p-3 text-left ${open ? "border-primary bg-primary/5" : "border-border bg-surface"}`}>
        <div className="flex items-center justify-between gap-2">
          <span className="truncate text-sm font-semibold">{t.name}</span>
          <Pill tone={t.is_research ? "research" : "neutral"}>{t.is_research ? "Research" : ROLE_LABEL[t.my_role]}</Pill>
        </div>
        <p className="mt-1 text-xs text-muted">
          {SEVERITY_TEXT[t.drift_severity ?? "unknown"] ?? t.drift_severity} · {t.activities ?? 0} activities
          {!t.is_research && <> · {t.members ?? 0} members · synced {shortDate(t.last_synced)}</>}
        </p>
      </button>
    </li>
  )
}

export default function TeamsPage() {
  const { user, teams, team, selectTeam, refreshTeams } = useSession()
  const router = useRouter()
  const [detail, setDetail] = useState<TeamDetail | null>(null)
  const [error, setError] = useState("")
  const [msg, setMsg] = useState("")
  const [newTeam, setNewTeam] = useState({ name: "", repo_url: "" })
  const [code, setCode] = useState("")
  const [supervisorCode, setSupervisorCode] = useState("")
  const [identities, setIdentities] = useState("")
  const [repo, setRepo] = useState("")
  const [busy, setBusy] = useState(false)
  const examiner = user?.role === "examiner"
  const supervisor = user?.role === "supervisor"
  const student = user?.role === "student"

  const loadDetail = useCallback(async (id: string) => {
    const d = await api<TeamDetail>(`/app/teams/${encodeURIComponent(id)}`)
    setDetail(d)
    setRepo(d.repo_url ?? "")
    setIdentities((d.members.find((m) => m.is_me)?.git_identities ?? []).join("\n"))
  }, [])

  useEffect(() => {
    refreshTeams().catch((e) => setError(e.message))
  }, [refreshTeams])

  useEffect(() => {
    if (team) loadDetail(team.team_id).catch((e) => setError(e.message))
    else setDetail(null)
  }, [team, loadDetail])

  const run = async (fn: () => Promise<unknown>, ok: string) => {
    setBusy(true)
    setError("")
    setMsg("")
    try {
      await fn()
      setMsg(ok)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const create = () => run(async () => {
    const t = await api<Team>("/app/teams", { method: "POST", json: newTeam })
    await refreshTeams()
    selectTeam(t.team_id)
    setNewTeam({ name: "", repo_url: "" })
  }, "Team created. Share the join code below with your teammates and supervisor.")

  const join = () => run(async () => {
    const t = await api<Team>("/app/teams/join", { method: "POST", json: { join_code: code } })
    await refreshTeams()
    selectTeam(t.team_id)
    setCode("")
  }, "You joined the team.")

  const addSupervisor = () => run(async () => {
    await api(`/app/teams/${team!.team_id}/supervisors`, { method: "POST", json: { supervisor_code: supervisorCode } })
    await loadDetail(team!.team_id)
    setSupervisorCode("")
  }, "Supervisor added.")

  const removeMember = (userId: string, name: string | null) => {
    if (!window.confirm(`Remove ${name ?? "this member"} from the team?`)) return
    run(async () => {
      await api(`/app/teams/${team!.team_id}/members/${encodeURIComponent(userId)}`, { method: "DELETE" })
      await Promise.all([loadDetail(team!.team_id), refreshTeams()])
    }, "Member removed.")
  }

  const saveIdentities = () => run(async () => {
    await api(`/app/teams/${team!.team_id}/me`, { method: "PUT", json: { git_identities: identities.split(/[\n,]/) } })
    await loadDetail(team!.team_id)
  }, "Saved. Run a sync (Behavior tab) so your commits are credited to you.")

  const linkAuthor = (userId: string, identity: string) => run(async () => {
    await api(`/app/teams/${team!.team_id}/link`, { method: "POST", json: { user_id: userId, identity } })
    await loadDetail(team!.team_id)
  }, "Linked. Run a sync (Behavior tab) so those commits are credited to that member.")

  const saveRepo = () => run(async () => {
    await api(`/app/teams/${team!.team_id}`, { method: "PUT", json: { repo_url: repo } })
    await Promise.all([loadDetail(team!.team_id), refreshTeams()])
  }, "Repository link saved.")

  const mine = teams.filter((t) => !t.is_research && t.my_role !== "panel" && !t.archived)
  const research = teams.filter((t) => t.is_research)
  const supervisedList = supervisorLayout.researchInGroupList && isSupervisorLayout(user?.role) ? [...mine, ...research] : mine
  const me = detail?.members.find((m) => m.is_me)
  const canEditRepo = detail && !detail.is_research && (detail.my_role === "lead" || detail.my_role === "supervisor")
  const canLink = canEditRepo
  const linkable = detail?.members.filter((m) => m.user_id && m.member_code) ?? []

  return (
    <div className="flex flex-col gap-4">
      {error && <Notice tone="error">{error}</Notice>}
      {msg && <Notice tone="success">{msg}</Notice>}

      {student && (
        <Card title="Join a team" note="Got a code from your team lead? Enter it here.">
          <div className="flex gap-2">
            <input className={inputClass} placeholder="e.g. H79-KN5" value={code} onChange={(e) => setCode(e.target.value)} aria-label="Join code" />
            <Button onClick={join} disabled={busy || !code.trim()}>Join</Button>
          </div>
        </Card>
      )}

      {supervisor && (
        <SupervisedGroups teams={supervisedList} openId={team?.team_id} onChanged={refreshTeams} onError={setError}
                          onOpen={(id) => {
                            selectTeam(id)
                            if (isSupervisorLayout(user?.role)) router.push(supervisorLayout.openGroupAt)
                          }} />
      )}

      {supervisor && <PanelReview onError={setError} />}

      {!supervisor && (
      <Card title={examiner ? "Current teams (anonymised)" : "My teams"}
            note={examiner
              ? "Every real team, read-only. Names, emails, repo links and chat are hidden; members appear as codes."
              : mine.length ? "Tap a team to open it." : "No teams yet."}>
        {mine.length > 0 && (
          <ul className="flex flex-col gap-2">
            {mine.map((t) => <TeamRow key={t.team_id} t={t} open={team?.team_id === t.team_id} onOpen={() => selectTeam(t.team_id)} />)}
          </ul>
        )}
      </Card>
      )}

      {detail && !isSupervisorLayout(user?.role) && (
        <Card title={detail.name} note={detail.is_research ? detail.label : detail.anonymised ? "Anonymised, read-only" : `Your role: ${ROLE_LABEL[detail.my_role]}`}>
          <div className="flex flex-col gap-4">
            {detail.join_code && (
              <div className="flex items-center justify-between rounded-lg border border-border bg-background p-3">
                <div>
                  <p className="text-xs text-muted">Join code: share it with your teammates</p>
                  <p className="font-mono text-lg font-semibold tracking-wider">{detail.join_code}</p>
                </div>
                <Button variant="secondary" onClick={() => navigator.clipboard?.writeText(detail.join_code!)}>Copy</Button>
              </div>
            )}

            {!detail.is_research && !detail.anonymised && (
              <Field label="GitHub repository" hint={canEditRepo ? "Public repos work directly; private ones need the server's GITHUB_TOKEN to have access." : undefined}>
                {canEditRepo ? (
                  <div className="flex gap-2">
                    <input className={inputClass} placeholder="https://github.com/owner/repo" value={repo} onChange={(e) => setRepo(e.target.value)} />
                    <Button variant="secondary" onClick={saveRepo} disabled={busy}>Save</Button>
                  </div>
                ) : (
                  <span className="text-sm text-muted">{detail.repo_url ?? "Not set yet (the team lead adds it)"}</span>
                )}
              </Field>
            )}

            <div>
              <h3 className="text-sm font-semibold">Members</h3>
              <ul className="mt-2 flex flex-col divide-y divide-border">
                {detail.members.map((m) => (
                  <li key={m.user_id ?? m.member_code} className="flex items-center justify-between gap-2 py-2 text-sm">
                    <span className="min-w-0 truncate">{m.name ?? <span className="font-mono">{m.member_code}</span>}{m.is_me && " (you)"}</span>
                    <span className="flex flex-none items-center gap-2">
                      {!detail.is_research && !detail.anonymised && <Pill>{ROLE_LABEL[m.team_role]}</Pill>}
                      {m.team_role !== "supervisor" && !detail.is_research && !detail.anonymised && (
                        <span className="text-xs text-muted">{m.git_identities.length ? `${m.git_identities.length} git id${m.git_identities.length > 1 ? "s" : ""}` : "no git id"}</span>
                      )}
                      {detail.my_role === "lead" && m.team_role === "member" && m.user_id && (
                        <Button variant="ghost" className="px-1 py-0 text-xs text-danger" disabled={busy}
                                onClick={() => removeMember(m.user_id!, m.name)}>Remove</Button>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            </div>

            {detail.unlinked_authors.length > 0 && (
              <Notice>
                <strong>Commits not linked to a member yet:</strong>{" "}
                {detail.unlinked_authors.map((a) => `${a.name} <${a.email}> (${a.commits})`).join(", ")}.
                {" "}If one of these is you, add that name or email below and sync again.
              </Notice>
            )}

            {!detail.is_research && (detail.my_role === "lead" || detail.my_role === "member") && (
              <Field label="Add your supervisor" hint="Ask your supervisor for their supervisor code.">
                <div className="flex gap-2">
                  <input className={inputClass} placeholder="e.g. 4QX-72B" value={supervisorCode}
                         onChange={(e) => setSupervisorCode(e.target.value)} />
                  <Button variant="secondary" onClick={addSupervisor} disabled={busy || !supervisorCode.trim()}>Add</Button>
                </div>
              </Field>
            )}

            {me && me.team_role !== "supervisor" && (
              <Field label="My git names / emails" hint="One per line. Use what your commits show (git config user.name / user.email, or your GitHub noreply email). Your account name and email are always included.">
                <textarea className={`${inputClass} h-20 font-mono`} value={identities} onChange={(e) => setIdentities(e.target.value)} />
                <div><Button variant="secondary" onClick={saveIdentities} disabled={busy}>Save git identities</Button></div>
              </Field>
            )}

            <div className="flex gap-2">
              <Button onClick={() => router.push("/behavior")}>Open behaviour dashboard</Button>
              {(detail.is_research || detail.my_role === "lead" || detail.my_role === "member") && (
                <Button variant="secondary" onClick={() => router.push("/chat")}>Open chat</Button>
              )}
            </div>
          </div>
        </Card>
      )}

      {student && (
        <Card title="Create a team" note="You become the team lead and get a join code to share with your teammates.">
          <div className="flex flex-col gap-3">
            <Field label="Team name">
              <input className={inputClass} value={newTeam.name} onChange={(e) => setNewTeam({ ...newTeam, name: e.target.value })} />
            </Field>
            <Field label="GitHub repository (optional, can be added later)">
              <input className={inputClass} placeholder="https://github.com/owner/repo" value={newTeam.repo_url}
                     onChange={(e) => setNewTeam({ ...newTeam, repo_url: e.target.value })} />
            </Field>
            <div><Button onClick={create} disabled={busy || newTeam.name.trim().length < 2}>Create team</Button></div>
          </div>
        </Card>
      )}

      {research.length > 0 && !isSupervisorLayout(user?.role) && (
        <Card title="Research dataset (Phase 1)"
              note="16 anonymised public GitHub team repositories with simulated chat. The evaluation results (Kappa, F1, drift test) come from these. Read-only.">
          <ul className="flex flex-col gap-2">
            {research.map((t) => <TeamRow key={t.team_id} t={t} open={team?.team_id === t.team_id} onOpen={() => selectTeam(t.team_id)} />)}
          </ul>
        </Card>
      )}

      {supervisor && <PastGroups />}
      {supervisor && <SupervisorCode />}
    </div>
  )
}
