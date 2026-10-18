// A member's name (real teams) next to their anonymous code, or just the code (research teams).
import styles from "./c2.module.css";

export default function MemberName({ code, names }: { code: string; names: Record<string, string> }) {
  return names[code]
    ? <span><strong>{names[code]}</strong> <span className={styles.mono}>{code}</span></span>
    : <span className={styles.mono}>{code}</span>;
}
