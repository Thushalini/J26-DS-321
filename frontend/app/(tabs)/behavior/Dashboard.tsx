"use client";
// The supervisor view for one team (FR09). It only displays data; page.tsx fetches it.
import { useMemo } from "react";

import PlotlyChart from "./PlotlyChart";
import MemberName from "./MemberName";
import RolesCard from "./RolesCard";
import styles from "./c2.module.css";
import {
  SEVERITY_STYLE, STAGE_LABEL, roleHistoryText, stageBreakdownChart,
} from "./charts";
import type { DriftDimension, Severity, Summary } from "./types";

function SeverityBadge({ severity, big = false }: { severity: Severity; big?: boolean }) {
  const s = SEVERITY_STYLE[severity] ?? SEVERITY_STYLE.unknown;
  return (
    <span className={`${styles.badge} ${big ? styles.heroBadge : ""}`}>
      <span className={styles.dot} style={{ background: s.color }} aria-hidden="true" />
      <span aria-hidden="true">{s.icon}</span>
      {s.label}
    </span>
  );
}

function DriftList({ dimensions }: { dimensions: DriftDimension[] }) {
  return (
    <ul className={styles.driftList}>
      {dimensions.map((d) => (
        <li key={d.dimension} className={styles.driftItem}>
          <div className={styles.driftHead}>
            <span className={styles.dimName}>{d.dimension}</span>
            <SeverityBadge severity={d.severity} />
          </div>
          <p className={styles.planLine}>
            {d.part && <>From <strong>{d.part}</strong> · </>}
            Planned: <strong>{d.planned ?? "–"}</strong> · Seen most: <strong>{d.seen_top ?? "–"}</strong>
            {d.drift_score !== null && <> · Drift score: <strong>{d.drift_score.toFixed(2)}</strong></>}
          </p>
          <p className={styles.why}>{d.explanation}</p>
        </li>
      ))}
    </ul>
  );
}

const pct = (x: number | undefined) => `${Math.round((x ?? 0) * 100)}%`;

export default function Dashboard({ summary, names = {}, simulatedChat = true, version = 0 }: {
  summary: Summary;
  names?: Record<string, string>;   // member code -> name (real teams only)
  simulatedChat?: boolean;
  version?: number;
}) {
  const mode = "light"; // the app shell is light; charts.ts also supports "dark" if the app adds it
  const stageChart = useMemo(
    () => stageBreakdownChart(summary.stage_breakdown, summary.lagging_stages, mode),
    [summary, mode],
  );
  const parts = summary.drift.parts ?? [];


  return (
    <div className={styles.grid}>
      <section className={styles.card} aria-labelledby="drift-title">
        <h2 id="drift-title" className={styles.cardTitle}>Plan drift</h2>
        <p className={styles.cardNote}>Is the team still working on the planned target, algorithm and metric?</p>
        <div className={styles.hero}>
          <SeverityBadge severity={summary.drift.overall_severity} big />
        </div>
        {parts.length > 0 && (
          <p className={styles.why}>
            This project has {parts.length} parts, each with its own plan. Below: the worst result per decision.
          </p>
        )}
        <DriftList dimensions={summary.drift.dimensions} />
      </section>

      {parts.map((p) => (
        <section key={p.part} className={styles.card} aria-label={`Plan drift: ${p.part}`}>
          <h2 className={styles.cardTitle}>Plan drift · {p.part}</h2>
          <p className={styles.cardNote}>Work in this part&apos;s folder, compared with this part&apos;s own plan.</p>
          <DriftList dimensions={p.dimensions} />
        </section>
      ))}

      <section className={styles.card} aria-labelledby="stage-title">
        <h2 id="stage-title" className={styles.cardTitle}>Where the team&apos;s work went</h2>
        <p className={styles.cardNote}>
          Share of commits and notebook edits in each lifecycle stage, {summary.window.start.slice(0, 10)} to{" "}
          {summary.window.end.slice(0, 10)}. Stages under 5% are marked lagging.
        </p>
        <PlotlyChart data={stageChart.data} layout={stageChart.layout} ariaLabel="Bar chart of work share per lifecycle stage" />
      </section>

      <RolesCard teamId={summary.team_id} names={names} version={version} />

      <section className={styles.card} aria-labelledby="roles-title">
        <h2 id="roles-title" className={styles.cardTitle}>Role changes over time and team talk</h2>
        <p className={styles.cardNote}>
          Current role = biggest smoothed stage share. Chat measures follow Group Communication Analysis
          {simulatedChat ? " (chat data is simulated for these public repositories)." : " (from the in-app team chat)."}
        </p>
        <ul className={styles.memberList}>
          {summary.members.map((m) => (
            <li key={m.member_id} className={styles.memberItem}>
              <div className={styles.memberHead}>
                <MemberName code={m.member_id} names={names} />
                <span className={styles.roleName}>
                  {m.dominant_role ? STAGE_LABEL[m.dominant_role] ?? m.dominant_role : "No role yet"}
                </span>
              </div>
              <p className={styles.why}>Role history: {roleHistoryText(m.role_trend)}</p>
              <dl className={styles.stats}>
                <div><dt>Participation</dt><dd>{pct(m.chat?.participation)}</dd></div>
                <div><dt>Responsiveness</dt><dd>{(m.chat?.responsiveness ?? 0).toFixed(2)}</dd></div>
                <div><dt>Social impact</dt><dd>{(m.chat?.social_impact ?? 0).toFixed(2)}</dd></div>
              </dl>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}