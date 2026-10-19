"use client";
// Behavior tab (Component 2): open http://localhost:3000/behavior
// The team comes from the header's team picker; data goes through /app (access-checked).
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui";
import { api, memberNames, shortDate, type SyncStatus, type TeamDetail } from "@/lib/api";
import { useSession } from "@/lib/session";
import Dashboard from "./Dashboard";
import WhoIsWho from "./WhoIsWho";
import styles from "./c2.module.css";
import type { Summary } from "./types";

function SyncPanel({ teamId, canSync, onDone }: { teamId: string; canSync: boolean; onDone: () => void }) {
  const [status, setStatus] = useState<SyncStatus | null>(null);
  const [error, setError] = useState("");
  const wasRunning = useRef(false);

  const poll = useCallback(async () => {
    const s = await api<SyncStatus>(`/app/teams/${encodeURIComponent(teamId)}/sync`);
    setStatus(s);
    if (wasRunning.current && s.status !== "running") onDone();
    wasRunning.current = s.status === "running";
  }, [teamId, onDone]);

  useEffect(() => {
    wasRunning.current = false;
    poll().catch((e) => setError(e.message));
  }, [poll]);

  useEffect(() => {
    if (status?.status !== "running") return;
    const id = setInterval(() => poll().catch(() => {}), 3000);
    return () => clearInterval(id);
  }, [status?.status, poll]);

  const start = async () => {
    setError("");
    try {
      setStatus(await api<SyncStatus>(`/app/teams/${encodeURIComponent(teamId)}/sync`, { method: "POST" }));
      wasRunning.current = true;
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const running = status?.status === "running";
  return (
    <section className={styles.card} aria-live="polite">
      <div className={styles.driftHead}>
        <h2 className={styles.cardTitle}>Repository sync</h2>
        {canSync && <Button onClick={start} disabled={running}>{running ? "Syncing…" : "Sync now"}</Button>}
      </div>
      <p className={styles.cardNote}>
        Last synced: {shortDate(status?.last_synced)}.{" "}
        {status?.status === "error" ? <span className={styles.error}>Last sync failed: {status.message}</span> : status?.message}
      </p>
      {error && <p className={styles.error} role="alert">{error}</p>}
    </section>
  );
}

export default function BehaviorPage() {
  const { team, refreshTeams } = useSession();
  const [days, setDays] = useState(30);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [detail, setDetail] = useState<TeamDetail | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [version, setVersion] = useState(0);         // bumped after a sync to reload everything

  const teamId = team?.team_id;

  useEffect(() => {
    if (!teamId) return;
    api<TeamDetail>(`/app/teams/${encodeURIComponent(teamId)}`).then(setDetail).catch(() => setDetail(null));
  }, [teamId, version]);

  useEffect(() => {
    if (!teamId) return;
    setLoading(true);
    setError("");
    api<Summary>(`/app/teams/${encodeURIComponent(teamId)}/behavior/summary?days=${days}`)
      .then(setSummary)
      .catch((e: Error) => {
        setSummary(null);
        setError(e.message);
      })
      .finally(() => setLoading(false));
  }, [teamId, days, version]);

  const onSynced = useCallback(() => {
    setVersion((v) => v + 1);
    refreshTeams().catch(() => {});
  }, [refreshTeams]);

  if (!team) return null;
  const showWho = detail && !detail.is_research && !detail.anonymised
    && (detail.my_role === "lead" || detail.my_role === "supervisor");

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <h1 className={styles.title}>Behavior</h1>
        <p className={styles.subtitle}>Lifecycle stages, plan drift and member roles · {team.name}</p>
        <div className={styles.filters}>
          <label>
            Window
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              <option value={14}>Last 14 days</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
            </select>
          </label>
        </div>
      </header>

      {!team.is_research && <SyncPanel teamId={team.team_id} canSync={team.my_role !== "viewer"} onDone={onSynced} />}
      {showWho && <WhoIsWho detail={detail} />}

      {error && <p className={styles.state} role="status">{error}</p>}
      {!error && loading && <p className={styles.state}>Loading…</p>}
      {!error && !loading && summary && (
        <Dashboard summary={summary} names={memberNames(detail)} simulatedChat={team.is_research} version={version} />
      )}
    </div>
  );
}
