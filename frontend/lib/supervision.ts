export type SupervisionStatus = "attention" | "fine" | "good"

export const supervisionStatuses: { id: SupervisionStatus; label: string; dot: string }[] = [
  { id: "attention", label: "Needs attention", dot: "bg-status-critical" },
  { id: "fine", label: "Going fine", dot: "bg-status-warning" },
  { id: "good", label: "Doing well", dot: "bg-status-good" },
]

export const unratedStatus = { label: "Not rated yet", dot: "bg-border" }

export function statusInfo(id: SupervisionStatus | null | undefined) {
  return supervisionStatuses.find((s) => s.id === id) ?? unratedStatus
}

export const supervisorCopy = {
  groupsTitle: "My groups",
  groupsNote: "Tap a group to open its analytics. The dot is your own status for the group.",
  noGroups: "No groups yet. Give students your supervisor code (bottom of this page) so they can add you.",
  menuLabel: "Group options",
  rename: "Rename",
  save: "Save",
  setStatus: "Set status",
  clearStatus: "Clear status",
  remove: "Remove from my list",
  removeConfirm: "Remove this group from your list? It moves to Past groups at the bottom of this page.",
  panelTitle: "Panel review",
  panelNote: "Reviewing another team? Ask for its join code. You can view one team at a time; entering the next code closes the previous one.",
  panelPlaceholder: "Team join code, e.g. H79-KN5",
  panelStart: "View",
  panelOpen: "Open analytics",
  panelEnd: "Finish review",
  panelCurrent: "Now reviewing",
  panelUntil: "Access ends",
  codeTitle: "My supervisor code",
  codeNote: "Students add you to their group with this code. Keep it private.",
  codeShow: "Show my supervisor code",
  codeCopy: "Copy",
  codeRegenerate: "Make a new code",
  codeRegenerateConfirm: "Make a new code? The old one stops working. Groups that already added you stay in your list.",
  codeRegenerated: "New code ready. Groups that already added you are still in your list.",
  pastTitle: "Past groups",
  pastNote: "Groups you removed from your list.",
  restore: "Back to my list",
  searchLabel: "Search groups",
  searchPlaceholder: "Search by group name",
  noMatches: "No groups match your search.",
} as const

export const supervisorLayout = {
  groupListAsPicker: true,
  tabs: ["home", "behavior", "summary"],
  landing: "/teams",
  openGroupAt: "/behavior",
  allowedPaths: ["/teams", "/behavior", "/summary", "/account", "/guide", "/feedback"],
} as const

export const supervisorHeaderCopy = {
  currentGroup: "Group",
  noGroup: "No group open",
  myGroups: "My groups",
  researchLabel: "Research dataset",
  researchPlaceholder: "Research dataset (Phase 1)…",
} as const
