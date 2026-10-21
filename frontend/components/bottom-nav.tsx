"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { navItemsFor } from "@/lib/navigation"
import { useSession } from "@/lib/session"

export function BottomNav() {
  const pathname = usePathname()
  const { user } = useSession()

  return (
    <nav className="sticky bottom-0 border-t border-border bg-surface">
      <ul className="flex">
        {navItemsFor(user?.role).map((item) => {
          const active = pathname.startsWith(item.href)
          return (
            <li key={item.id} className="flex-1">
              <Link
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={`block py-3 text-center text-xs ${active ? "font-semibold text-primary" : "text-muted"}`}
              >
                {item.label}
              </Link>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
