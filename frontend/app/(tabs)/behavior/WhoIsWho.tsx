// "Who's who": which member code is which person (code + name only, no emails or git identities).
// For everyone in a real team (lead, members) and its supervisors; research teams and the examiner
// view stay codes-only (NFR07).
import type { TeamDetail } from "@/lib/api";
import styles from "./c2.module.css";

export default function WhoIsWho({ detail }: { detail: TeamDetail }) {
  const linked = detail.members.filter((m) => m.member_code);
  const unlinked = detail.unlinked_authors;

  return (
    <section className={styles.card} aria-labelledby="who-title">
      <h2 id="who-title" className={styles.cardTitle}>Who&apos;s who</h2>
      <p className={styles.cardNote}>
        Each code is one team member. Only this team and its supervisors see the names.
      </p>
      <div className={styles.tableWrap}>
        <table className={styles.whoTable}>
          <thead>
            <tr><th scope="col">Member code</th><th scope="col">Name</th></tr>
          </thead>
          <tbody>
            {linked.map((m) => (
              <tr key={m.member_code}>
                <td className={styles.mono}>{m.member_code}</td>
                <td>{m.name}{m.is_me && " (you)"}</td>
              </tr>
            ))}
            {unlinked.map((a) => (
              <tr key={a.member_code}>
                <td className={styles.mono}>{a.member_code}</td>
                <td><span className={styles.error}>Not linked to a member yet</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {unlinked.length > 0 && (
        <p className={styles.why}>
          Link an unlinked author to the right member on the Teams page, then sync again.
        </p>
      )}
    </section>
  );
}
