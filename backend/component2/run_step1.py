from component2.config import RAW_DATA_DIR
from component2.db import connect, save_activities
from component2.load_github import load_repo


def main():
    conn = connect()
    repo_dirs = sorted(d for d in RAW_DATA_DIR.iterdir() if (d / "commits.json").exists())
    print(f"Found {len(repo_dirs)} repos in {RAW_DATA_DIR}")

    for repo_dir in repo_dirs:
        commit_rows, cell_rows = load_repo(repo_dir)
        new = save_activities(conn, commit_rows + cell_rows)
        members = len({r["member_id"] for r in commit_rows})
        print(f"  {repo_dir.name:<45} commits={len(commit_rows):<5} cells={len(cell_rows):<5} "
              f"members={members:<3} new={new}")

    rows = conn.execute("SELECT source, COUNT(*) FROM activities GROUP BY source").fetchall()
    print("\nDatabase totals:", dict(rows))


if __name__ == "__main__":
    main()