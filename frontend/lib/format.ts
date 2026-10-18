import { site } from "@/lib/site"

const dateFormatter = new Intl.DateTimeFormat(site.locale, { dateStyle: "medium" })

export function formatPercent(value: number): string {
  return `${Math.round(value)}%`
}

export function formatSigned(value: number): string {
  const sign = value > 0 ? "+" : value < 0 ? "−" : ""
  return `${sign}${Math.abs(value)}`
}

export function formatDate(iso: string): string {
  return dateFormatter.format(new Date(iso))
}
