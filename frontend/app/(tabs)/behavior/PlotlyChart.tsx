"use client";
// Draws a Plotly chart in the browser only (Plotly needs `window`, so it is loaded on demand).
import { useEffect, useRef } from "react";

type Props = { data: unknown[]; layout: Record<string, unknown>; ariaLabel: string };

export default function PlotlyChart({ data, layout, ariaLabel }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    const el = ref.current;
    import("plotly.js-dist-min").then((mod) => {
      const Plotly = mod.default ?? mod;
      if (!cancelled && el) {
        Plotly.react(el, data, layout, { displayModeBar: false, responsive: true });
      }
    });
    return () => {
      cancelled = true;
    };
  }, [data, layout]);

  return <div ref={ref} role="img" aria-label={ariaLabel} style={{ width: "100%" }} />;
}