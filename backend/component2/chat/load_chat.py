"""
Step 4b: Import team chat CSV files into the activities table (FR03).

Expected CSV columns: msg_id, timestamp, sender, message, reply_to
(one file per team in component2/data/chat/, file name = team_id).
Works for the simulated chat AND for a real chat exported to this format:
real names are turned into member codes and scrubbed from the text (NFR07).
Rows with a missing or wrong timestamp, or an empty message, are skipped and reported
(api/contract.py), so one bad line doesn't stop the whole file.

Run from backend:
    python -m component2.chat.load_chat
"""
import csv
import re

from component2.anonymise import MemberResolver
from component2.api.contract import split_valid
from component2.config import CHAT_DIR
from component2.db import connect, save_activities

MEMBER_CODE = re.compile(r"M-[0-9a-f]{6}")      # already anonymised (simulated chat uses these)


def load_file(path):
    team_id = path.stem
    resolver = MemberResolver(team_id)
    with open(path, newline="", encoding="utf-8-sig") as f:
        messages = list(csv.DictReader(f))

    rows = []
    for m in messages:
        sender = (m.get("sender") or "").strip()
        if not sender or not m.get("msg_id"):
            continue
        member_id = sender if MEMBER_CODE.fullmatch(sender) else resolver.get_id(sender)
        reply_to = f"chat:{team_id}:{m['reply_to']}" if m.get("reply_to") else None
        rows.append({
            "activity_id": f"chat:{team_id}:{m['msg_id']}",
            "team_id": team_id,
            "source": "chat",
            "member_id": member_id,
            "ts": (m.get("timestamp") or "").strip(),
            "text": m.get("message") or "",
            "meta": {"reply_to": reply_to, "file": path.name},
        })
    rows, rejected = split_valid(rows)
    skipped = len(messages) - len(rows) - len(rejected)
    for r in rows:                       # scrub after all names are known
        r["text"] = resolver.scrub(r["text"])
    return rows, rejected + [(None, "missing msg_id or sender")] * skipped


def main():
    conn = connect()
    files = sorted(CHAT_DIR.glob("*.csv"))
    print(f"Found {len(files)} chat files in {CHAT_DIR}")
    for path in files:
        rows, rejected = load_file(path)
        new = save_activities(conn, rows)
        print(f"  {path.stem:<45} messages={len(rows):<5} new={new}  skipped={len(rejected)}")
        for activity_id, reason in rejected[:5]:
            print(f"      skipped {activity_id or '-'}: {reason}")
    total = conn.execute("SELECT COUNT(*) FROM activities WHERE source = 'chat'").fetchone()[0]
    print(f"Chat messages in database: {total}")


if __name__ == "__main__":
    main()