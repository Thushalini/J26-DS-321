import { samplePrediction } from "@/lib/predict/sample-data"
import type { Prediction } from "@/lib/predict/types"

export async function getPrediction(): Promise<Prediction> {
  return samplePrediction
}
