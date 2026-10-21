// A member's name (real teams) next to their anonymous code. Without a name the code is labelled
// "Member", so nobody mistakes an M-xxxxxx code for something other than a person in the team.
import styles from "./c2.module.css";

export function memberLabel(code: string, names: Record<string, string>) {
  return names[code] ? `${names[code]} (${code})` : `Member ${code}`;
}

export default function MemberName({ code, names }: { code: string; names: Record<string, string> }) {
  return names[code]
    ? <span><strong>{names[code]}</strong> <span className={styles.mono}>{code}</span></span>
    : <span>Member <span className={styles.mono}>{code}</span></span>;
}
