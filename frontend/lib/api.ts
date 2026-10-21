// Calls to the backend. Every request carries the signed-in user's token.
import type { SupervisionStatus } from "@/lib/supervision"

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"

const TOKEN_KEY = "buddy.token"

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function setToken(token: string | null) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    // private mode: the session just won't survive a reload
  }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

export async function api<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, headers, ...rest } = init
  const token = getToken()
  let res: Response
  try {
    res = await fetch(`${API}${path}`, {
      ...rest,
      body: json === undefined ? rest.body : JSON.stringify(json),
      headers: {
        ...(json === undefined ? {} : { "Content-Type": "application/json" }),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...headers,
      },
    })
  } catch {
    throw new ApiError(0, `Can't reach the backend at ${API}. Is "uvicorn main:app" running?`)
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (typeof body.detail === "string") detail = body.detail
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ")
    } catch {}
    throw new ApiError(res.status, detail)
  }
  const type = res.headers.get("content-type") ?? ""
  return (type.includes("application/json") ? res.json() : res.text()) as Promise<T>
}

// ---- Shapes returned by /app ----

export type UserRole = "student" | "supervisor" | "examiner"
export type TeamRole = "lead" | "member" | "supervisor" | "viewer" | "panel"

export interface User {
  user_id: string
  email: string
  name: string
  role: UserRole
  supervisor_code?: string
}

export interface Team {
  team_id: string
  name: string
  repo_url: string | null
  is_research: boolean
  anonymised?: boolean               // real team seen by an examiner: member codes only, no personal data
  label: string | null
  my_role: TeamRole
  join_code?: string
  sync_status: "idle" | "running" | "done" | "error" | null
  sync_message: string | null
  last_synced: string | null
  activities?: number
  members?: number
  drift_severity?: string
  supervision_status?: SupervisionStatus | null
  archived?: boolean
}

export interface TeamMember {
  user_id?: string
  name: string | null
  email?: string
  team_role: TeamRole
  member_code: string | null
  git_identities: string[]
  is_me?: boolean
}

export interface UnlinkedAuthor {
  member_code: string
  name: string
  email: string
  commits: number
}

export interface TeamDetail extends Omit<Team, "members"> {
  members: TeamMember[]
  unlinked_authors: UnlinkedAuthor[]
}

export interface SyncStatus {
  status: "idle" | "running" | "done" | "error"
  message: string | null
  started: string | null
  last_synced: string | null
}

export const ROLE_LABEL: Record<string, string> = {
  student: "Student",
  supervisor: "Supervisor",
  examiner: "Examiner",
  lead: "Team lead",
  member: "Member",
  viewer: "Viewer (read-only)",
  panel: "Panel review (temporary)",
}

/** member code -> display name, for the people in a real team */
export function memberNames(detail: TeamDetail | null): Record<string, string> {
  const names: Record<string, string> = {}
  for (const m of detail?.members ?? []) if (m.member_code && m.name) names[m.member_code] = m.name
  return names
}

export function shortDate(ts: string | null | undefined) {
  if (!ts) return "never"
  const d = new Date(ts)
  return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })
}
// ---- Component 1: the living blueprint (nine planning fields) ----

export type PlanStatus = "missing" | "uncertain" | "known" | "confirmed" | "conflicting" | string

export interface PlanEvidence {
  speaker: string
  text: string
  source_id: string
  source_type: string
  time: string | null
}

export interface PlanField {
  name: string
  label: string
  status: PlanStatus
  value: string | null
  confidence_band: "High" | "Medium" | "Low" | null
  mentions: number
  candidates?: string[]
  evidence?: PlanEvidence[]
}

export interface PlanData {
  available: true
  can_see_sources: boolean
  team_id: string
  version: number
  updated_at: string | null
  readiness: {
    complete: boolean
    confirmed: number
    total: number
    missing: string[]
    conflicting: string[]
    not_confirmed: string[]
    low_confidence: string[]
  }
  next_gaps: string[]
  fields: PlanField[]
  note: string
}

export type PlanResponse = PlanData | { available: false; reason: string }

export function getPlan(teamId: string) {
  return api<PlanResponse>(`/app/teams/${encodeURIComponent(teamId)}/plan`)
}