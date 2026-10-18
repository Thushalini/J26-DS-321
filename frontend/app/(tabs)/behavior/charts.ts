// Turns API data into Plotly chart settings. No React here, so it is easy to test.
import type { Member, Severity, WindowMember } from "./types";

export const STAGES = [
  "data_cleaning", "eda", "feature_engineering", "model_development",
  "model_evaluation", "deployment_mlops", "documentation",
];

export const STAGE_LABEL: Record<string, string> = {
  data_cleaning: "Data cleaning",
  eda: "EDA",
  feature_engineering: "Feature engineering",
  model_development: "Model development",
  model_evaluation: "Model evaluation",
  deployment_mlops: "Deployment / MLOps",
  documentation: "Documentation",
};

// Colours: one fixed colour per stage, in lifecycle order (validated for colour-blind safety),
// with separate steps for light and dark mode.
const SERIES = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
};

export const THEME = {
  light: { text: "#0b0b0b", muted: "#52514e", grid: "#e4e3df", surface: "#fcfcfb" },
  dark: { text: "#f0efec", muted: "#c3c2b7", grid: "#383835", surface: "#1a1a19" },
};

export type Mode = "light" | "dark";

export function stageColor(stage: string, mode: Mode): string {
  return SERIES[mode][STAGES.indexOf(stage)] ?? THEME[mode].muted;
}

// Severity is shown with colour + icon + word, never colour alone
export const SEVERITY_STYLE: Record<Severity, { color: string; icon: string; label: string }> = {
  none: { color: "#0ca30c", icon: "✓", label: "On plan" },
  minor: { color: "#fab219", icon: "!", label: "Minor drift" },
  moderate: { color: "#ec835a", icon: "!!", label: "Moderate drift" },
  significant: { color: "#d03b3b", icon: "▲", label: "Significant drift" },
  no_plan: { color: "#8a8983", icon: "–", label: "No plan" },
  no_evidence: { color: "#8a8983", icon: "?", label: "No recent evidence" },
  unknown: { color: "#8a8983", icon: "?", label: "Unknown" },
};

function baseLayout(mode: Mode) {
  const t = THEME[mode];
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "inherit", color: t.text, size: 13 },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.grid, font: { color: t.text } },
  };
}

/** Chart 1: share of the team's recent work in each stage (one series, one colour). */
export function stageBreakdownChart(shares: Record<string, number>, lagging: string[], mode: Mode) {
  const t = THEME[mode];
  const stages = [...STAGES].reverse(); // first stage at the top
  const values = stages.map((s) => shares[s] ?? 0);
  const labels = stages.map((s) => `  ${Math.round((shares[s] ?? 0) * 100)}%${lagging.includes(s) ? "  · lagging" : ""}`);
  return {
    data: [{
      type: "bar",
      orientation: "h",
      y: stages.map((s) => STAGE_LABEL[s]),
      x: values,
      marker: { color: SERIES[mode][0] },
      text: labels,
      textposition: "outside",
      textfont: { color: t.muted },
      cliponaxis: false,
      hovertemplate: "%{y}: %{x:.0%} of recent work<extra></extra>",
    }],
    layout: {
      ...baseLayout(mode),
      height: 300,
      margin: { l: 150, r: 80, t: 10, b: 30 },
      bargap: 0.45,
      xaxis: { tickformat: ".0%", gridcolor: t.grid, zeroline: false, range: [0, Math.max(0.1, ...values) * 1.25] },
      yaxis: { automargin: true },
      showlegend: false,
    },
  };
}

/** Chart 2: each member's mix of stages (100% stacked bars, one colour per stage). */
export function memberMixChart(members: Member[], mode: Mode) {
  const t = THEME[mode];
  const names = members.map((m) => m.member_id);
  const data = STAGES.map((stage) => ({
    type: "bar",
    orientation: "h",
    name: STAGE_LABEL[stage],
    y: names,
    x: members.map((m) => m.stage_mix?.[stage] ?? 0),
    marker: { color: stageColor(stage, mode), line: { color: t.surface, width: 2 } }, // 2px gap between segments
    hovertemplate: `%{y}<br>${STAGE_LABEL[stage]}: %{x:.0%}<extra></extra>`,
  }));
  return {
    data,
    layout: {
      ...baseLayout(mode),
      barmode: "stack",
      height: Math.max(220, 110 + members.length * 40),
      margin: { l: 10, r: 10, t: 10, b: 30 },
      bargap: 0.35,
      xaxis: { tickformat: ".0%", range: [0, 1], gridcolor: t.grid, zeroline: false },
      yaxis: { automargin: true },
      legend: { orientation: "h", y: -0.2, yanchor: "top", traceorder: "normal", font: { color: t.muted, size: 11 } },
    },
  };
}

/** Role history as short text, e.g. "2026-W36 EDA → 2026-W38 Model development". */
export function roleHistoryText(trend: Member["role_trend"]): string {
  const changes: string[] = [];
  let last: string | null | undefined = undefined;
  for (const { week, dominant } of trend) {
    if (dominant && dominant !== last) {
      changes.push(`${week} ${STAGE_LABEL[dominant] ?? dominant}`);
      last = dominant;
    }
  }
  return changes.length ? changes.join(" → ") : "–";
}

/** Chart 3: every member's mix of ALL stages in the chosen window (code + chat combined). */
export function windowMixChart(members: WindowMember[], mode: Mode) {
  const t = THEME[mode];
  const names = members.map((m) => m.member_id);
  const data = STAGES.map((stage) => ({
    type: "bar",
    orientation: "h",
    name: STAGE_LABEL[stage],
    y: names,
    x: members.map((m) => m.combined_mix?.[stage] ?? 0),
    marker: { color: stageColor(stage, mode), line: { color: t.surface, width: 2 } },
    hovertemplate: `%{y}<br>${STAGE_LABEL[stage]}: %{x:.0%}<extra></extra>`,
  }));
  return {
    data,
    layout: {
      ...baseLayout(mode),
      barmode: "stack",
      height: Math.max(220, 110 + members.length * 40),
      margin: { l: 10, r: 10, t: 10, b: 30 },
      bargap: 0.35,
      xaxis: { tickformat: ".0%", range: [0, 1], gridcolor: t.grid, zeroline: false },
      yaxis: { automargin: true },
      legend: { orientation: "h", y: -0.2, yanchor: "top", traceorder: "normal", font: { color: t.muted, size: 11 } },
    },
  };
}

/** "Model development 60%, EDA 30%" - the top stages of a mix, as text. */
export function topStagesText(mix: Record<string, number>, limit = 3): string {
  const top = STAGES.filter((s) => (mix[s] ?? 0) > 0)
    .sort((a, b) => mix[b] - mix[a])
    .slice(0, limit)
    .map((s) => `${STAGE_LABEL[s]} ${Math.round(mix[s] * 100)}%`);
  return top.length ? top.join(", ") : "–";
}