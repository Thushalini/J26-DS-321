import { redirect } from "next/navigation"
import { defaultNavItem } from "@/lib/navigation"

export default function Page() {
  redirect(defaultNavItem.href)
}
