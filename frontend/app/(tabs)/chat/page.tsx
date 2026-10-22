"use client"
import type React from "react"
// Team chat. Real teams: members' messages become Component 2 activities (half weight in roles,
// replies feed responsiveness / social impact). Research teams: the simulated chat, read-only.
// Buddy (Component 1) messages arrive in the same list with sender_type "assistant".
import { useCallback, useEffect, useRef, useState } from "react"

import { Button, Notice, Pill, inputClass } from "@/components/ui"
import { api, shortDate, type Team } from "@/lib/api"
import { useSession } from "@/lib/session"

interface Message {
  msg_id: string
  sender: string
  member_code: string | null
  ts: string
  text: string
  reply_to: string | null
  mine: boolean
  team_role: string | null
  sender_type?: "member" | "assistant"
}

interface Chat {
  read_only: boolean
  simulated: boolean
  messages: Message[]
}

const POLL_MS = 5000

/** Real teams' chat is for the team lead and members only (not supervisors or examiners). */
function chatHidden(team: Team) {
  return !team.is_research && team.my_role !== "lead" && team.my_role !== "member"
}

export default function ChatPage() {
  const { team } = useSession()
  const [chat, setChat] = useState<Chat | null>(null)
  const [text, setText] = useState("")
  const [replyTo, setReplyTo] = useState<Message | null>(null)
  const [error, setError] = useState("")
  const [sending, setSending] = useState(false)
  const bottom = useRef<HTMLDivElement>(null)
  const count = useRef(0)

  const load = useCallback(async () => {
    if (!team || chatHidden(team)) return
    const c = await api<Chat>(`/app/teams/${encodeURIComponent(team.team_id)}/chat`)
    setChat(c)
    setError("")
  }, [team])

  useEffect(() => {
    setChat(null)
    count.current = 0
    load().catch((e) => setError(e.message))
    if (!team || team.is_research || chatHidden(team)) return
    const id = setInterval(() => load().catch(() => {}), POLL_MS)
    return () => clearInterval(id)
  }, [team, load])

  useEffect(() => {
    if (chat && chat.messages.length !== count.current) {
      count.current = chat.messages.length
      bottom.current?.scrollIntoView({ block: "end" })
    }
  }, [chat])

  const send = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!team || !text.trim()) return
    setSending(true)
    try {
      await api(`/app/teams/${encodeURIComponent(team.team_id)}/chat`, {
        method: "POST", json: { text, reply_to: replyTo?.msg_id ?? null },
      })
      setText("")
      setReplyTo(null)
      await load()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSending(false)
    }
  }

  if (!team) return null
  if (chatHidden(team)) {
    return (
      <div className="flex flex-col gap-3">
        <h1 className="text-lg font-semibold">Team chat</h1>
        <Notice>
          A team&apos;s chat is private: only the team lead and members can read it. How much of the chat goes to each
          lifecycle stage is still included in the Behavior tab.
        </Notice>
      </div>
    )
  }
  const byId = new Map(chat?.messages.map((m) => [m.msg_id, m]))

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Team chat</h1>
        {chat?.simulated && <Pill tone="research">Simulated chat</Pill>}
      </div>
      {!team.is_research && (
        <p className="text-xs text-muted">Your messages are analysed for lifecycle stage and team interaction (shown on the Behavior tab). Buddy listens too and keeps the planning blueprint up to date.</p>
      )}
      {error && <Notice tone="error">{error}</Notice>}
      {!chat && !error && <p className="text-sm text-muted">Loading...</p>}
      {chat && chat.messages.length === 0 && <Notice>No messages yet. Say hello and post your first update.</Notice>}

      <ul className="flex flex-col gap-2">
        {chat?.messages.map((m) => {
          const isBuddy = m.sender_type === "assistant"
          const parent = m.reply_to ? byId.get(m.reply_to) : undefined
          const bubble = isBuddy
            ? "border-border border-l-4 border-l-primary bg-primary/5"
            : m.mine
              ? "border-primary/30 bg-primary/10"
              : "border-border bg-surface"
          return (
            <li key={m.msg_id} className={`flex ${m.mine && !isBuddy ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[85%] rounded-card border px-3 py-2 ${bubble}`}>
                <p className="text-xs text-muted">
                  {isBuddy ? (
                    <span className="mr-1 rounded-full bg-primary px-2 py-0.5 text-xs font-semibold text-primary-foreground">Buddy</span>
                  ) : (
                    <span className={chat.simulated ? "font-mono" : "font-semibold text-foreground"}>{m.mine ? "You" : m.sender}</span>
                  )}
                  {m.team_role === "supervisor" && " · supervisor"} · {shortDate(m.ts)}
                </p>
                {parent && <p className="mt-1 border-l-2 border-border pl-2 text-xs text-muted">↪ {parent.text.slice(0, 80)}</p>}
                <p className="mt-1 whitespace-pre-wrap text-sm">{m.text}</p>
                {!chat.read_only && !isBuddy && (
                  <button type="button" className="mt-1 text-xs text-primary" onClick={() => setReplyTo(m)}>Reply</button>
                )}
              </div>
            </li>
          )
        })}
      </ul>
      <div ref={bottom} />

      {chat && !chat.read_only && (
        <form onSubmit={send} className="sticky bottom-14 flex flex-col gap-2 rounded-card border border-border bg-surface p-2">
          {replyTo && (
            <p className="flex items-center justify-between text-xs text-muted">
              <span className="truncate">Replying to {replyTo.mine ? "yourself" : replyTo.sender}: {replyTo.text.slice(0, 50)}</span>
              <button type="button" className="text-primary" onClick={() => setReplyTo(null)}>Cancel</button>
            </p>
          )}
          <div className="flex gap-2">
            <input className={inputClass} placeholder="Post an update..." value={text} onChange={(e) => setText(e.target.value)} maxLength={2000} aria-label="Message" />
            <Button type="submit" disabled={sending || !text.trim()}>Send</Button>
          </div>
        </form>
      )}
    </div>
  )
}