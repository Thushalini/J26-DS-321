import type React from "react"
import type { Metadata } from "next"
import { SessionProvider } from "@/lib/session"
import { site } from "@/lib/site"
import "./globals.css"

export const metadata: Metadata = {
  title: site.name,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <SessionProvider>{children}</SessionProvider>
      </body>
    </html>
  )
}
