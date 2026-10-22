// Shapes of the JSON returned by the Component 2 API (/c2/teams/{team_id}/summary)

export type Severity = "none" | "minor" | "moderate" | "significant" | "no_plan" | "no_evidence" | "unknown";

export interface DriftDimension {
  dimension: "target" | "algorithm" | "metric";
  planned: string | null;
  seen_top: string | null;
  drift_score: number | null;
  severity: Severity;
  explanation: string;
  part?: string | null;      // multi-model projects: which part this result comes from (null = whole project)
}

export interface DriftPart {
  part: string;
  dimensions: DriftDimension[];
}

export interface Member {
  member_id: string;
  stage_mix: Record<string, number>;
  chat: { participation: number; responsiveness: number; social_impact: number };
  dominant_role: string | null;
  role_trend: { week: string; dominant: string | null }[];
}

export interface Summary {
  team_id: string;
  generated_at: string;
  window: { start: string; end: string };
  stage_breakdown: Record<string, number>;
  lagging_stages: string[];
  drift: { overall_severity: Severity; dimensions: DriftDimension[]; parts?: DriftPart[] };
  members: Member[];
}

export interface TeamInfo {
  team_id: string;
  activities: number;
  first_activity: string;
  last_activity: string;
}

export interface WindowMember {
  member_id: string;
  code_items: number;
  chat_items: number;
  combined_mix: Record<string, number>;
  code_mix: Record<string, number>;
  chat_mix: Record<string, number>;
  main_roles: string[];
  evidence: "none" | "low" | "ok";
  last_active: string | null;        // latest commit / notebook edit / chat message, any time
}

export interface WindowRoles {
  team_id: string;
  window: "3d" | "7d" | "30d" | "all";
  start: string;
  end: string;
  chat_weight: number;
  members: WindowMember[];
}