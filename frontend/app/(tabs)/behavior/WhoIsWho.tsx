// "Who's who": which member code is which person, and which git name/email each code came from.
// Only for the lead / supervisor of a real team; research teams and the examiner view stay codes-only (NFR07).
import type { TeamDetail } from "@/lib/api";
import styles from "./c2.module.css";

export default function WhoIsWho({ detail }: { detail: TeamDetail }) {
  const linked = detail.members.filter((m) => m.member_code);
  const unlinked = detail.unlinked_authors;

  return (
    <section className={styles.card} aria-labelledby="who-title">
      <h2 id="who-title" className={styles.cardTitle}>Who&apos;s who</h2>
      <p className={styles.cardNote}>
        Which member code belongs to which person. Only the team lead and supervisors see this.
      </p>
      <div className={styles.tableWrap}>
        <table className={styles.whoTable}>
          <thead>
            <tr><th scope="col">Code</th><th scope="col">Person</th><th scope="col">Git names / emails</th></tr>
          </thead>
          <tbody>
            {linked.map((m) => (
              <tr key={m.member_code}>
                <td className={styles.mono}>{m.member_code}</td>
                <td>{m.name}{m.is_me && " (you)"}</td>
                <td>{[m.email, ...m.git_identities].filter(Boolean).join(", ")}</td>
              </tr>
            ))}
            {unlinked.map((a) => (
              <tr key={a.member_code}>
                <td className={styles.mono}>{a.member_code}</td>
                <td><span className={styles.error}>Not linked</span> · {a.commits} commit{a.commits === 1 ? "" : "s"}</td>
                <td>{a.name} &lt;{a.email}&gt;</td>
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
