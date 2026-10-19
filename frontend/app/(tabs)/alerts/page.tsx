"use client";

import { useState } from "react";
import data from "@/lib/component3-alerts.json";

type Level = "HIGH" | "MEDIUM" | "LOW";

interface Component3Alert {
  id: string;
  run_id: string;
  generated_at: string;
  n_epochs: number;
  anomaly_score: number;
  severity_score: number;
  severity_level: Level;
  title: string;
  evidence: string | null;
  recommended_cause: string | null;
  second_cause: string | null;
  recommendation: string | null;
  root_cause_probabilities: Record<string, number>;
  status: string;
}

const CAUSE_LABELS: Record<string, string> = {
  overfitting: "Overfitting",
  stalled: "Stalled experiment",
  data_leakage: "Data leakage",
  resource_bottleneck: "Resource bottleneck",
  premature_evaluation: "Premature evaluation",
};

const LEVEL_STYLE: Record<Level, string> = {
  HIGH: "border-status-critical/40 bg-status-critical/10 text-status-critical",
  MEDIUM: "border-status-warning bg-status-warning/20 text-foreground",
  LOW: "border-status-good/40 bg-status-good/10 text-status-good",
};

const LEVEL_ORDER: Record<Level, number> = { HIGH: 0, MEDIUM: 1, LOW: 2 };

// Real outputs of the Component 3 pipeline on sample runs
// (created by backend/component3/scripts/generate_alerts.py).
const alerts = data.alerts as unknown as Component3Alert[];

function causeLabel(key: string | null): string {
  if (!key) return "None";
  return CAUSE_LABELS[key] ?? key;
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

function LevelBadge({ level }: { level: Level }) {
  return (
    <span
      className={`mt-0.5 shrink-0 rounded-bar border px-1.5 py-0.5 font-mono text-[10px] font-semibold ${LEVEL_STYLE[level]}`}
    >
      {level}
    </span>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 20 20"
      className={`mt-1 h-4 w-4 shrink-0 text-subtle transition-transform ${open ? "rotate-90" : ""}`}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M7 4l6 6-6 6" />
    </svg>
  );
}

function CauseBars({
  probs,
  top,
}: {
  probs: Record<string, number>;
  top: string | null;
}) {
  const rows = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  return (
    <ul className="space-y-1.5">
      {rows.map(([key, p]) => (
        <li key={key} className="flex items-center gap-2 text-xs">
          <span className="w-32 shrink-0 text-muted">{causeLabel(key)}</span>
          <span className="h-2 flex-1 overflow-hidden rounded-bar bg-chart-track">
            <span
              className={`block h-full rounded-bar ${key === top ? "bg-chart-series" : "bg-chart-axis"}`}
              style={{ width: `${Math.round(p * 100)}%` }}
            />
          </span>
          <span className="w-9 text-right tabular-nums text-foreground">
            {Math.round(p * 100)}%
          </span>
        </li>
      ))}
    </ul>
  );
}

function StatCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: string;
}) {
  return (
    <div className="rounded-card border border-border bg-surface p-3">
      <p className="font-mono text-[10px] tracking-wide text-subtle">{label}</p>
      <p className={`mt-1 font-mono text-2xl font-semibold ${tone}`}>{value}</p>
    </div>
  );
}

export default function Page() {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [resolvedIds, setResolvedIds] = useState<string[]>([]);

  const issues = alerts
    .filter((a) => a.severity_level !== "LOW")
    .sort(
      (a, b) =>
        LEVEL_ORDER[a.severity_level] - LEVEL_ORDER[b.severity_level] ||
        b.severity_score - a.severity_score,
    );
  const healthy = alerts.filter((a) => a.severity_level === "LOW");

  const isResolved = (id: string) => resolvedIds.includes(id);
  const ordered = [
    ...issues.filter((a) => !isResolved(a.id)),
    ...issues.filter((a) => isResolved(a.id)),
  ];
  const openCount = issues.filter((a) => !isResolved(a.id)).length;
  const resolvedCount = issues.length - openCount;

  const toggleResolved = (id: string) =>
    setResolvedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );

  return (
    <div className="space-y-4">
      <header className="space-y-1">
        <p className="font-mono text-[10px] tracking-wide text-subtle">
          COMPONENT 3 · ADAPTIVE INTERVENTION
        </p>
        <h1 className="text-lg font-semibold text-foreground">
          Anomaly Detection
        </h1>
        <p className="text-sm text-muted">
          Experiment signals are scored by the Component 3 pipeline. It flags
          the issue, names the likely root cause and suggests an intervention.
        </p>
        <p className="text-xs text-subtle">
          Showing the pipeline&rsquo;s outputs on sample runs.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-3">
        <StatCard label="OPEN" value={openCount} tone="text-status-critical" />
        <StatCard
          label="RESOLVED"
          value={resolvedCount}
          tone="text-status-good"
        />
      </div>

      <ul className="space-y-3">
        {ordered.map((a) => {
          const open = expandedId === a.id;
          const done = isResolved(a.id);
          return (
            <li
              key={a.id}
              className={`rounded-card border border-border bg-surface ${done ? "opacity-60" : ""}`}
            >
              <button
                type="button"
                onClick={() => setExpandedId(open ? null : a.id)}
                aria-expanded={open}
                className="flex w-full items-start gap-3 p-3 text-left"
              >
                <LevelBadge level={a.severity_level} />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm font-semibold text-foreground">
                    {a.title}
                  </span>
                  <span className="mt-0.5 block text-xs text-subtle">
                    {formatTime(a.generated_at)}
                    {done ? " · resolved" : ""}
                  </span>
                </span>
                <Chevron open={open} />
              </button>

              {open && (
                <div className="space-y-3 border-t border-border p-3">
                  {a.evidence && (
                    <p className="text-sm text-muted">{a.evidence}</p>
                  )}
                  <p className="text-xs text-subtle">
                    Anomaly score {a.anomaly_score.toFixed(2)} · Severity{" "}
                    {a.severity_score.toFixed(2)} · {a.n_epochs} epochs
                  </p>

                  <div className="rounded-card border border-border bg-background p-3">
                    <p className="font-mono text-[10px] tracking-wide text-subtle">
                      LIKELY ROOT CAUSE
                    </p>
                    <p className="mt-1 text-sm font-semibold text-status-critical">
                      {causeLabel(a.recommended_cause)}
                    </p>
                    {a.second_cause && (
                      <p className="text-xs text-muted">
                        Also possible: {causeLabel(a.second_cause)}
                      </p>
                    )}
                    <div className="mt-3">
                      <CauseBars
                        probs={a.root_cause_probabilities}
                        top={a.recommended_cause}
                      />
                    </div>
                  </div>

                  {a.recommendation && (
                    <div className="rounded-card border border-primary/30 bg-primary/5 p-3">
                      <p className="font-mono text-[10px] tracking-wide text-subtle">
                        SUGGESTED INTERVENTION
                      </p>
                      <p className="mt-1 text-sm text-foreground">
                        {a.recommendation}
                      </p>
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => toggleResolved(a.id)}
                    className={
                      done
                        ? "w-full rounded-card border border-border px-3 py-2 text-sm font-semibold text-foreground"
                        : "w-full rounded-card bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground"
                    }
                  >
                    {done ? "Reopen" : "Mark resolved"}
                  </button>
                </div>
              )}
            </li>
          );
        })}
      </ul>

      {healthy.length > 0 && (
        <section aria-labelledby="healthy-heading" className="space-y-2">
          <h2
            id="healthy-heading"
            className="font-mono text-[10px] tracking-wide text-subtle"
          >
            HEALTHY RUNS · NO ACTION NEEDED
          </h2>
          <ul className="divide-y divide-border rounded-card border border-border bg-surface">
            {healthy.map((a) => (
              <li
                key={a.id}
                className="flex items-center justify-between gap-3 p-3 text-sm"
              >
                <span className="flex items-center gap-2">
                  <span
                    className={`rounded-bar border px-1.5 py-0.5 font-mono text-[10px] font-semibold ${LEVEL_STYLE.LOW}`}
                  >
                    OK
                  </span>
                  <span className="text-foreground">{a.run_id}</span>
                </span>
                <span className="text-xs text-muted">
                  anomaly score {a.anomaly_score.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
