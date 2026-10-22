"""Check that Component 1 can use the database in backend/.env (Supabase or the local file).

Run from the backend folder:
    python -m component1.db_check

What it does, in order, and it stops at the first problem:
  1. says which database it will use (never prints the address or the password)
  2. connects
  3. creates Component 1's own tables if they are missing (all start with c1_; no team table is changed)
  4. checks that the app tables Component 1 reads exist (chat_messages, team_members, users, blueprints)
  5. writes one test row and ROLLS IT BACK, so nothing is left behind
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
APP_TABLES = ("chat_messages", "team_members", "users", "blueprints")
C1_TABLES = ("c1_events", "c1_agent_messages", "c1_processed", "c1_documents", "c1_doc_chunks", "c1_doc_facts")


def _describe(url: str) -> str:
    if not url:
        return "the LOCAL SQLite file (DATABASE_URL is empty)"
    host = url.split("@", 1)[-1].split("/", 1)[0]
    port = host.rsplit(":", 1)[-1] if ":" in host else "5432"
    kind = "Supabase" if "supabase" in host else "PostgreSQL"
    note = ""
    if port == "6543":
        note = "  WARNING: port 6543 is the Transaction pooler; use the Session pooler (port 5432) instead."
    return f"{kind} (port {port}){note}"


def _exists(conn, table: str) -> bool:
    try:
        conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchall()
        return True
    except Exception:
        rollback = getattr(getattr(conn, "raw", None), "rollback", None)
        if rollback:
            rollback()
        return False


def main() -> int:
    from component1.extraction.gemini_provider import load_env_file

    load_env_file(BACKEND_DIR / ".env")
    url = (os.environ.get("DATABASE_URL") or "").strip()
    print(f"1. Database: {_describe(url)}")

    from component2.db import connect
    from component1.storage import ensure_c1_tables

    start = time.perf_counter()
    try:
        conn = connect(create_tables=False)
    except Exception as exc:
        print(f"2. Connect: FAILED ({type(exc).__name__}). Check the address, password and network.")
        return 1
    print(f"2. Connect: OK in {time.perf_counter() - start:.1f} s")

    try:
        ensure_c1_tables(conn)
        missing = [t for t in C1_TABLES if not _exists(conn, t)]
        if missing:
            print(f"3. Component 1 tables: MISSING {missing}")
            return 1
        print(f"3. Component 1 tables: OK ({len(C1_TABLES)} tables)")

        absent = [t for t in APP_TABLES if not _exists(conn, t)]
        if absent:
            print(f"4. App tables: not found yet: {absent}. They are created when the app first starts "
                  "(uvicorn main:app). Component 1 works once they exist.")
        else:
            print("4. App tables: OK")

        conn.execute("INSERT INTO c1_processed (team_id, message_id, processed_at) VALUES (?, ?, ?)",
                     ("__c1_check__", "__c1_check__", "2000-01-01T00:00:00Z"))
        seen = conn.execute("SELECT COUNT(*) FROM c1_processed WHERE team_id = ?", ("__c1_check__",)).fetchone()[0]
        conn.raw.rollback()                       # never committed: nothing is left behind
        left = conn.execute("SELECT COUNT(*) FROM c1_processed WHERE team_id = ?", ("__c1_check__",)).fetchone()[0]
        if seen == 1 and left == 0:
            print("5. Write test: OK (one row written and rolled back)")
        else:
            print(f"5. Write test: unexpected result (seen {seen}, left {left})")
            return 1
    finally:
        conn.close()
    print("\nComponent 1 is ready to use this database. Restart the backend to start saving here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
