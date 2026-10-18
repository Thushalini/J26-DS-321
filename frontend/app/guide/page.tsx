"use client"
// Explains each stakeholder's flow. Open it during a demo: http://localhost:3000/guide
import Link from "next/link"
import { useState } from "react"

import { Card, Pill } from "@/components/ui"
import { useSession } from "@/lib/session"

type Step = { title: string; where: string; text: string }

const FLOWS: Record<string, { who: string; goal: string; steps: Step[] }> = {
  lead: {
    who: "Student - team lead",
    goal: "Set the team up once, so everyone's work is analysed.",
    steps: [
      { title: "Create an account", where: "Sign in → Create account → Student", text: "Use your university email." },
      { title: "Create the team", where: "Teams → Create a team", text: "Give the team a name and paste the GitHub repository link (https://github.com/owner/repo). You get a join code like 3F9A-C21B." },
      { title: "Share the join code", where: "Teams", text: "Send the code to your teammates and your supervisor. Each joins with it." },
      { title: "Enter the plan (blueprint)", where: "Blueprint tab", text: "Planned target variable, algorithm and evaluation metric. Component 1 will fill this automatically later; drift is measured against it." },
      { title: "Sync the repository", where: "Behavior tab → Sync now", text: "Buddy clones the repo, reads every commit and notebook cell, classifies each into a lifecycle stage, checks drift and updates roles (about 1-2 minutes)." },
      { title: "Fix unlinked authors", where: "Teams → Members", text: "If commits show up as 'unlinked', ask that teammate to add the name/email they commit with, then sync again." },
    ],
  },
  member: {
    who: "Student - team member",
    goal: "Get credit for your work and see where the team's effort goes.",
    steps: [
      { title: "Create an account and join", where: "Teams → Join with a code", text: "Type the code your lead shared." },
      { title: "Link your git identity", where: "Teams → My git names / emails", text: "Add the name and email in your commits (git config user.name / user.email, or your GitHub noreply email). This links your commits to you." },
      { title: "Talk in the team chat", where: "Chat tab", text: "Post updates and reply to teammates. Messages count (at half weight) towards your role, and replies feed the responsiveness and social-impact measures." },
      { title: "Check your role", where: "Behavior tab", text: "See your stage mix for 3 days, a week, a month or the whole project, and how your role changed over time." },
      { title: "Give feedback", where: "Give feedback button", text: "Does your role match what you actually did? Rate the screen and say what's wrong." },
    ],
  },
  supervisor: {
    who: "Supervisor",
    goal: "See all your teams at a glance and step in early.",
    steps: [
      { title: "Create an account", where: "Create account → Supervisor", text: "No code needed. Ask your team lead for the team's join code." },
      { title: "Join each team", where: "Teams → Join with a code", text: "Use the team's join code. Supervisors watch the team's behaviour; the team chat stays private to the students." },
      { title: "Overview", where: "Teams", text: "Every team with its drift headline, activity count and last sync. Research dataset teams are shown separately and marked as such." },
      { title: "Drill into a team", where: "Behavior tab", text: "Plan drift (target / algorithm / metric) with a one-line reason, work share per stage with lagging stages, and each member's roles." },
      { title: "Read feedback", where: "Feedback (header)", text: "Feedback from your teams (students stay anonymous) and your own, with average ratings." },
    ],
  },
  examiner: {
    who: "Examiner / researcher",
    goal: "Check the method on the Phase 1 research dataset and collect evaluation feedback.",
    steps: [
      { title: "Sign in", where: "Create account → Examiner (staff code) or the demo examiner account", text: "Read-only. You can't create, join or change teams." },
      { title: "Open a research team", where: "Team picker → Research dataset", text: "16 anonymised public GitHub repos with simulated chat, labelled as such. These are the teams the Kappa, F1 and drift results were measured on." },
      { title: "Inspect the outputs", where: "Blueprint / Chat / Behavior", text: "Inferred blueprint, simulated chat and the same Component 2 dashboard that real teams see." },
      { title: "Export feedback", where: "Feedback → Download CSV", text: "All feedback from every role, for the evaluation chapter." },
    ],
  },
}

export default function GuidePage() {
  const { user } = useSession()
  const [tab, setTab] = useState<keyof typeof FLOWS>(user?.role === "supervisor" ? "supervisor" : user?.role === "examiner" ? "examiner" : "lead")
  const flow = FLOWS[tab]

  return (
    <main className="mx-auto flex min-h-dvh max-w-app flex-col gap-4 p-4 pb-10">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">How Buddy.ds works</h1>
        <Link href={user ? "/teams" : "/login"} className="text-sm text-primary">{user ? "Back to app" : "Sign in"}</Link>
      </div>

      <Card title="The idea in one line">
        <p className="text-sm">
          The behavioural models were developed and evaluated on 16 anonymised public data-science repositories
          (<strong>Phase 1</strong>, research dataset). The same pipeline now runs for registered teams on their own
          repository, in-app chat and plan (<strong>Phase 2</strong>).
        </p>
        <ol className="mt-3 flex flex-col gap-1 text-sm text-muted">
          <li>1. Repo + chat → activities (commits, notebook cells, messages)</li>
          <li>2. Each activity → lifecycle stage (MiniLM + logistic regression, keyword baseline as fallback)</li>
          <li>3. Recent mentions vs blueprint → plan drift (target, algorithm, metric)</li>
          <li>4. Each member&apos;s stage mix + chat measures per week → roles and role shifts</li>
        </ol>
      </Card>

      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Stakeholder">
        {Object.entries(FLOWS).map(([key, f]) => (
          <button key={key} type="button" role="tab" aria-selected={tab === key} onClick={() => setTab(key)}
                  className={`rounded-full border px-3 py-1.5 text-xs font-medium ${tab === key ? "border-primary bg-primary text-primary-foreground" : "border-border bg-surface"}`}>
            {f.who}
          </button>
        ))}
      </div>

      <Card title={flow.who} note={flow.goal}>
        <ol className="flex flex-col gap-3">
          {flow.steps.map((s, i) => (
            <li key={s.title} className="flex gap-3">
              <span className="flex h-6 w-6 flex-none items-center justify-center rounded-full bg-primary text-xs font-semibold text-primary-foreground">{i + 1}</span>
              <div className="min-w-0">
                <p className="text-sm font-semibold">{s.title}</p>
                <Pill>{s.where}</Pill>
                <p className="mt-1 text-sm text-muted">{s.text}</p>
              </div>
            </li>
          ))}
        </ol>
      </Card>

      <Card title="Who can see what">
        <ul className="flex flex-col gap-2 text-sm">
          <li><strong>Your real team</strong>: its members and the supervisors who joined it. Names are shown inside the team.</li>
          <li><strong>Research dataset teams</strong>: supervisors and examiners only, read-only, anonymised member codes, simulated chat.</li>
          <li><strong>Feedback</strong>: examiners see all; supervisors see their teams&apos; feedback with students anonymous.</li>
        </ul>
      </Card>
    </main>
  )
}
