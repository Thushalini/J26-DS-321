"use client";
// Each member's roles (ALL stages, not just one) for 3 days, a week, a month or the whole project.
// Uses code AND chat, so a member with only 2-3 commits still shows what they contributed.
import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import MemberName from "./MemberName";
import PlotlyChart from "./PlotlyChart";
import styles from "./c2.module.css";
import { STAGE_LABEL, topStagesText, windowMixChart } from "./charts";
import type { WindowRoles } from "./types";

const WINDOWS = [
  { key: "3d", label: "3 days" },
  { key: "7d", label: "Week" },
  { key: "30d", label: "Month" },
  { key: "all", label: "Whole project" },
] as const;

export default function RolesCard({ teamId, names = {}, version = 0 }: { teamId: string; names?: Record<string, string>; version?: number }) {
  const [windowKey, setWindowKey] = useState<string>("7d");
  const [data, setData] = useState<WindowRoles | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setError("");
    api<WindowRoles>(`/app/teams/${encodeURIComponent(teamId)}/behavior/roles?window=${windowKey}`)
      .then((d) => setData(d))
      .catch((e: Error) => setError(e.message));
  }, [teamId, windowKey, version]);

  const chart = useMemo(
    () => (data ? windowMixChart(data.members.map((m) => ({ ...m, member_id: names[m.member_id] ?? m.member_id })), "light") : null),
    [data, names],
  );
  const active = data?.members.filter((m) => m.evidence !== "none") ?? [];
  const idle = data?.members.filter((m) => m.evidence === "none") ?? [];

  return (
    <section className={styles.card} aria-labelledby="roles-window-title">
      <h2 id="roles-window-title" className={styles.cardTitle}>Each member&apos;s roles</h2>
      <p className={styles.cardNote}>
        Share of each member&apos;s work in every stage, from commits, notebook edits and chat
        (a chat message counts half as much as a commit).
      </p>

      <div className={styles.segmented} role="group" aria-label="Time window">
        {WINDOWS.map((w) => (
          <button key={w.key} type="button" aria-pressed={windowKey === w.key}
                  className={windowKey === w.key ? styles.segmentOn : styles.segment}
                  onClick={() => setWindowKey(w.key)}>
            {w.label}
          </button>
        ))}
      </div>

      {error && <p className={styles.error} role="alert">{error}</p>}
      {data && (
        <>
          <p className={styles.why}>{data.start.slice(0, 10)} to {data.end.slice(0, 10)}</p>
          {active.length > 0 && chart && (
            <PlotlyChart data={chart.data} layout={chart.layout} ariaLabel="Stacked bar chart of each member's work per stage in the chosen window" />
          )}
          <ul className={styles.memberList}>
            {active.map((m) => (
              <li key={m.member_id} className={styles.memberItem}>
                <div className={styles.memberHead}>
                  <MemberName code={m.member_id} names={names} />
                  <span className={styles.roleName}>{m.main_roles.map((s) => STAGE_LABEL[s] ?? s).join(" + ")}</span>
                </div>
                <p className={styles.why}>
                  From {m.code_items} commit/notebook item{m.code_items === 1 ? "" : "s"} and {m.chat_items} chat
                  message{m.chat_items === 1 ? "" : "s"}{m.evidence === "low" ? " · little evidence, treat as a hint" : ""}
                </p>
                <p className={styles.planLine}>Code: {topStagesText(m.code_mix)}</p>
                <p className={styles.planLine}>Chat: {topStagesText(m.chat_mix)}</p>
              </li>
            ))}
          </ul>
          {idle.length > 0 && (
            <p className={styles.why}>No activity in this window: {idle.map((m) => names[m.member_id] ?? m.member_id).join(", ")}</p>
          )}
        </>
      )}
    </section>
  );
}