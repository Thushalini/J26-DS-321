"use client"
// Who is signed in, which teams they can see, and which team is open right now.
import type React from "react"
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react"

import { api, ApiError, setToken, getToken, type Team, type User } from "@/lib/api"

const TEAM_KEY = "buddy.team"

interface Session {
  ready: boolean
  user: User | null
  teams: Team[]
  team: Team | null
  selectTeam: (teamId: string) => void
  refreshTeams: () => Promise<Team[]>
  signIn: (token: string, user: User) => void
  signOut: () => Promise<void>
}

const SessionContext = createContext<Session | null>(null)

function savedTeam() {
  try {
    return localStorage.getItem(TEAM_KEY)
  } catch {
    return null
  }
}

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false)
  const [user, setUser] = useState<User | null>(null)
  const [teams, setTeams] = useState<Team[]>([])
  const [teamId, setTeamId] = useState<string | null>(null)

  const refreshTeams = useCallback(async () => {
    const list = await api<Team[]>("/app/teams")
    setTeams(list)
    setTeamId((current) => {
      const wanted = current ?? savedTeam()
      return list.some((t) => t.team_id === wanted) ? wanted : (list.find((t) => !t.is_research) ?? list[0])?.team_id ?? null
    })
    return list
  }, [])

  useEffect(() => {
    if (!getToken()) {
      setReady(true)
      return
    }
    api<User>("/app/auth/me")
      .then((u) => {
        setUser(u)
        return refreshTeams()
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) setToken(null)     // only a rejected token signs you out
      })
      .finally(() => setReady(true))
  }, [refreshTeams])

  const selectTeam = useCallback((id: string) => {
    setTeamId(id)
    try {
      localStorage.setItem(TEAM_KEY, id)
    } catch {}
  }, [])

  const signIn = useCallback(
    (token: string, u: User) => {
      setToken(token)
      setUser(u)
      refreshTeams().catch(() => {})
    },
    [refreshTeams],
  )

  const signOut = useCallback(async () => {
    await api("/app/auth/logout", { method: "POST" }).catch(() => {})
    setToken(null)
    setUser(null)
    setTeams([])
    setTeamId(null)
  }, [])

  const value = useMemo<Session>(
    () => ({
      ready,
      user,
      teams,
      team: teams.find((t) => t.team_id === teamId) ?? null,
      selectTeam,
      refreshTeams,
      signIn,
      signOut,
    }),
    [ready, user, teams, teamId, selectTeam, refreshTeams, signIn, signOut],
  )

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}

export function useSession() {
  const s = useContext(SessionContext)
  if (!s) throw new Error("useSession must be used inside <SessionProvider>")
  return s
}
