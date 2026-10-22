// A Buddy (Component 1) chat message, split into labelled blocks so it is easy to read.
// Buddy's text is one line per part (backend/component1/agent/policy.py): an optional
// "Noted: a = x (...); b = y (...)." line, then at most one question or warning.
// Lines that do not match a known shape are shown as they are.

import type React from "react"

type Tone = "saved" | "conflict" | "check" | "found" | "question" | "confirm" | "clarify" | "plain"

const LOOK: Record<Tone, { label: string; cls: string; chip: string }> = {
  saved: { label: "Saved", cls: "", chip: "bg-status-good/15 text-status-good" },
  conflict: { label: "Team disagrees", cls: "border-status-critical/40 bg-status-critical/5", chip: "bg-status-critical/15 text-status-critical" },
  check: { label: "Check", cls: "border-status-warning bg-status-warning/10", chip: "bg-status-warning/25 text-foreground" },
  found: { label: "From your documents", cls: "border-primary/40 bg-primary/5", chip: "bg-primary/15 text-primary" },
  question: { label: "Question", cls: "border-primary/40 bg-surface", chip: "bg-primary/15 text-primary" },
  confirm: { label: "Please confirm", cls: "border-status-warning bg-status-warning/10", chip: "bg-status-warning/25 text-foreground" },
  clarify: { label: "Needs detail", cls: "border-primary/40 bg-primary/5", chip: "bg-primary/15 text-primary" },
  plain: { label: "", cls: "", chip: "" },
}

interface SavedItem { field: string; value: string; note: string }

function savedItems(line: string): SavedItem[] {
  const body = line.replace(/^Noted:\s*/, "").replace(/\.$/, "")
  return body.split("; ").map((part) => {
    const eq = part.indexOf(" = ")
    if (eq < 0) return { field: "", value: part, note: "" }
    const field = part.slice(0, eq)
    let value = part.slice(eq + 3)
    let note = ""
    const open = value.lastIndexOf(" (")
    if (open >= 0 && value.endsWith(")")) {
      note = value.slice(open + 2, -1)
      value = value.slice(0, open)
    }
    return { field, value, note }
  })
}

function noteTone(note: string) {
  if (/conflicts|rejected/.test(note)) return "text-status-critical"
  if (/confirmed|finalized/.test(note)) return "text-status-good"
  if (/uncertain/.test(note)) return "text-foreground"
  return "text-muted"
}

function Block({ tone, title, children }: { tone: Tone; title?: string; children: React.ReactNode }) {
  const look = LOOK[tone]
  return (
    <div className={`rounded-bar ${look.cls ? `border px-2 py-1.5 ${look.cls}` : ""}`}>
      <p className="flex flex-wrap items-center gap-1.5">
        <span className={`rounded-full px-1.5 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide ${look.chip}`}>
          {look.label}
        </span>
        {title && <span className="text-xs font-semibold">{title}</span>}
      </p>
      <div className="mt-1 text-sm">{children}</div>
    </div>
  )
}

function Line({ line }: { line: string }) {
  if (line.startsWith("Noted: ")) {
    return (
      <Block tone="saved">
        <ul className="flex flex-col gap-1">
          {savedItems(line).map((it, i) => (
            <li key={i}>
              {it.field && <span className="text-xs text-muted">{it.field}: </span>}
              <span className="font-semibold">{it.value}</span>
              {it.note && <span className={`block text-xs ${noteTone(it.note)}`}>{it.note}</span>}
            </li>
          ))}
        </ul>
      </Block>
    )
  }
  let m = line.match(/^Possible problem with the (.+?): (.*?)\s*Is that intended\?$/)
  if (m) {
    return (
      <Block tone="check" title={m[1]}>
        <p>{m[2]}</p>
        <p className="mt-1 font-medium">Is that intended?</p>
      </Block>
    )
  }
  m = line.match(/^I saw different answers for the (.+?): (.*?)\s*(Which one should we go with\?)$/)
  if (m) {
    return (
      <Block tone="conflict" title={m[1]}>
        <p>{m[2]}</p>
        <p className="mt-1 font-medium">{m[3]}</p>
      </Block>
    )
  }
  m = line.match(/^I found a possible (.+?) in (.+?): (.*?)\.\s*Should we use that\?\s*(\(.*\))?$/)
  if (m) {
    return (
      <Block tone="found" title={m[2]}>
        <p><span className="text-xs text-muted">{m[1]}: </span><span className="font-semibold">{m[3]}</span></p>
        <p className="mt-1 font-medium">Should we use that?</p>
        {m[4] && <p className="text-xs text-muted">{m[4].slice(1, -1)}</p>}
      </Block>
    )
  }
  m = line.match(/^Can someone confirm the (.+?): (.*?)\? I ask because (.*)\.$/)
  if (m) {
    return (
      <Block tone="confirm" title={m[1]}>
        <p><span className="font-semibold">{m[2]}</span>: can someone confirm it?</p>
        <p className="mt-1 text-xs text-muted">Why I ask: {m[3]}.</p>
      </Block>
    )
  }
  if (line.startsWith("A detail is missing: ")) {
    return <Block tone="clarify"><p>{line.slice("A detail is missing: ".length)}</p></Block>
  }
  if (line.startsWith("I noticed ")) {
    const cut = line.indexOf(" It matters because ")
    if (cut >= 0) {
      return (
        <Block tone="question">
          <p>{line.slice(0, cut)}</p>
          <p className="mt-1 text-xs text-muted">Why it matters: {line.slice(cut + " It matters because ".length)}</p>
        </Block>
      )
    }
    return <Block tone="question"><p>{line}</p></Block>
  }
  return <p className="text-sm">{line}</p>
}

export function BuddyMessage({ text }: { text: string }) {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean)
  return (
    <div className="mt-1 flex flex-col gap-2">
      {lines.map((l, i) => <Line key={i} line={l} />)}
    </div>
  )
}
