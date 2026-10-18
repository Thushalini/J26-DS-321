import { statusThresholds } from "@/lib/predict/config"
import type { StatusLevel } from "@/lib/predict/types"

export function getStatusLevel(score: number): StatusLevel {
  if (score >= statusThresholds.good) return "good"
  if (score >= statusThresholds.warning) return "warning"
  return "critical"
}
