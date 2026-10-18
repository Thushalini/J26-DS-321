import type React from "react"

type IconProps = { className?: string }

function Icon({ className, children }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      {children}
    </svg>
  )
}

export function IconCheck(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3 8.5l3.5 3.5L13 5" />
    </Icon>
  )
}

export function IconAlert(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M8 2.5l6 10.5H2z" />
      <path d="M8 7v2.5" />
    </Icon>
  )
}

export function IconCross(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 4l8 8M12 4l-8 8" />
    </Icon>
  )
}

export function IconArrowUp(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M8 13V3M4 7l4-4 4 4" />
    </Icon>
  )
}

export function IconArrowDown(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M8 3v10M4 9l4 4 4-4" />
    </Icon>
  )
}
